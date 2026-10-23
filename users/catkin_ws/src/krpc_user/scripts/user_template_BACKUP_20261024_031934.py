#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Kibo-RPC 7th User Node - OPTIMIZED STABLE & FAST EDITION

Optimasi:
1. Menggunakan built-in controller (/ctl/command) yang stabil, presisi, dan aman
   (tanpa osilasi liar atau tabrakan).
2. Parameter hardware dituning maksimal: v_max=0.25 m/s, f_max=0.48 N,
   w_max=0.5236 rad/s, t_max=0.019 Nm, dwell/duration_goal=0.20s.
3. In-flight Level 3 Handshake: Begitu robot masuk margin Lv3 (d < 0.088m, angle < 8.5 deg)
   selama 0.20s, langsung lapor CheckpointPassed tanpa menunggu action berhenti.
4. TSP Held-Karp rute optimal: Memilih urutan checkpoint dengan waktu tempuh minimum.
5. Timeout adaptif berbasis jarak (14s - 26s per CP) dengan fallback aman Lv2.
6. Kode bersih, terstruktur rapi, dan cepat dieksekusi.
"""
import json
import math
import threading
import rospy
import actionlib
import cv2
import numpy as np

try:
    import cv2.aruco as aruco
except ImportError:
    aruco = None

from platform_msgs.msg import UserLogic
from platform_msgs.srv import StopProcessingUserNode, StopProcessingUserNodeResponse
from geometry_msgs.msg import PoseStamped
from krpc_ib2_msgs.srv import MissionStart, MissionFinished, CheckPoints
from krpc_ib2_msgs.msg import CheckpointPassedAction, CheckpointPassedGoal
from ib2_msgs.msg import CtlCommandAction, CtlCommandGoal, CtlCommandResult, CtlStatusType

try:
    from ib2_msgs.msg import Navigation
except ImportError:
    Navigation = None

from sensor_msgs.msg import Image, CameraInfo


class UserTemplate:

    # ------------------------------------------------------------------
    # Target Level 3 & Toleransi
    # ------------------------------------------------------------------
    FAST_PASS_DIST  = 0.088   # < 0.10 m (Lv3 batas juri)
    FAST_PASS_ANGLE = 8.5     # < 10.0 deg (Lv3 batas juri)
    LENIENT_DIST    = 0.20    # fallback band Lv2
    LENIENT_ANGLE   = 20.0

    # ------------------------------------------------------------------
    # Timing & Validasi Kestabilan
    # ------------------------------------------------------------------
    VALIDATION_HOLD_SEC     = 0.20  # stabil di Lv3 selama 0.20s sebelum kirim event
    REQUIRED_STABLE_SAMPLES = 1
    AFTER_MOTION_WAIT_SEC   = 0.50

    TIME_LIMIT_SEC          = 300.0
    TIME_SAFETY_SEC         = 10.0  # cadangan untuk laporan misi akhir
    MOVE_TIMEOUT_BASE       = 8.0
    GOAL_TIMEOUT_SEC        = 35.0

    # Profil kinematika untuk TSP route planner
    MOVE_VMAX_MPS           = 0.25
    MOVE_AMAX_MPS2          = 0.05
    ROT_WMAX_DPS            = 30.0
    ROT_ALPHA_DPS2          = 40.0

    # ArUco detection
    ENABLE_ARUCO_SCAN       = True
    ARUCO_CAPTURE_WAIT_SEC  = 0.50
    MARKER_SIZE             = 0.02

    def __init__(self):
        rospy.init_node('krpc_user')

        self.__command = actionlib.SimpleActionClient('/ctl/command', CtlCommandAction)
        self.__checkpoint_client = actionlib.SimpleActionClient(
            '/event/checkpoint_passed', CheckpointPassedAction)

        self.__mission_start    = rospy.ServiceProxy('/event/mission_start', MissionStart)
        self.__mission_finished = rospy.ServiceProxy('/event/mission_finished', MissionFinished)
        self.__checkpoints      = rospy.ServiceProxy('/mission/get_checkpoints', CheckPoints)

        self.__nav_lock      = threading.Lock()
        self.__current_pos   = None
        self.__current_quat  = None
        self.__current_speed = 0.0
        self.__nav_seq       = 0

        if Navigation is not None:
            self.__sub_nav = rospy.Subscriber(
                '/sensor_fusion/navigation', Navigation, self.__callback_navigation, queue_size=1)

        self.__latest_image         = None
        self.__image_seq            = 0
        self.__camera_matrix        = None
        self.__dist_coeffs          = None
        self.__cam_lock             = threading.Lock()
        self.__aruco_lock           = threading.Lock()
        self.__aruco_worker_lock    = threading.Lock()
        self.__aruco_worker_running = False
        self.__aruco_seen           = {}
        self.__aruco_dict           = None
        self.__aruco_params         = None

        if self.ENABLE_ARUCO_SCAN and aruco is not None:
            self.__aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            self.__aruco_params = aruco.DetectorParameters_create()
            self.__sub_image = rospy.Subscriber(
                '/camera_main/image_raw', Image, self.__callback_image, queue_size=1)
            self.__sub_info = rospy.Subscriber(
                '/camera_main/camera_info', CameraInfo, self.__callback_camera_info, queue_size=1)

        self.__subscriber_start = rospy.Subscriber(
            '/ib2_user/start', UserLogic, self.__callback_start)
        self.__stop_processing_server = rospy.Service(
            '/ib2_user/stop', StopProcessingUserNode, self.__stop_processing)

        self.__started          = False
        self.__action_active    = False
        self.__controller_tuned = False
        self.__t_start          = None
        rospy.set_param('/ib2_user/ready', True)

    # ==================================================================
    # Callbacks
    # ==================================================================

    def __callback_navigation(self, msg):
        with self.__nav_lock:
            try:
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                self.__current_pos  = [float(p.x), float(p.y), float(p.z)]
                self.__current_quat = [float(o.x), float(o.y), float(o.z), float(o.w)]
                t = msg.twist.linear
                self.__current_speed = math.sqrt(t.x**2 + t.y**2 + t.z**2)
                self.__nav_seq += 1
            except Exception:
                pass

    def __callback_image(self, msg):
        if self.ENABLE_ARUCO_SCAN:
            try:
                with self.__cam_lock:
                    self.__latest_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        (msg.height, msg.width))
                self.__image_seq += 1
            except Exception:
                pass

    def __callback_camera_info(self, msg):
        if self.__camera_matrix is None:
            self.__camera_matrix = np.array(msg.K).reshape(3, 3)
            self.__dist_coeffs   = np.array(msg.D)

    def __get_nav(self):
        with self.__nav_lock:
            if self.__current_pos is None or self.__current_quat is None:
                return None
            return (list(self.__current_pos), list(self.__current_quat),
                    self.__current_speed, self.__nav_seq)

    # ==================================================================
    # Math & Evaluator (Identik dengan Juri)
    # ==================================================================

    def __dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def __normalize_quaternion(self, q):
        norm = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        if norm <= 1e-9:
            return [0.0, 0.0, 0.0, 1.0]
        return [q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm]

    def __rotate_vector_by_quaternion(self, vector, quaternion):
        qx, qy, qz, qw = self.__normalize_quaternion(quaternion)
        vx, vy, vz = vector
        t0 = qw*vx + qy*vz - qz*vy
        t1 = qw*vy - qx*vz + qz*vx
        t2 = qw*vz + qx*vy - qy*vx
        t3 = -qx*vx - qy*vy - qz*vz
        return [
            t0*qw + t3*(-qx) - t1*qz + t2*qy,
            t1*qw + t3*(-qy) + t0*qz - t2*qx,
            t2*qw + t3*(-qz) - t0*qy + t1*qx,
        ]

    def __normalize_vector(self, v):
        norm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if norm <= 1e-9:
            return [0.0, 0.0, 0.0]
        return [v[0]/norm, v[1]/norm, v[2]/norm]

    def __vector_angle_deg(self, v1, v2):
        dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]))
        return math.degrees(math.acos(dot))

    def __twist_angle_deg(self, quaternion, axis):
        qx, qy, qz, qw = self.__normalize_quaternion(quaternion)
        if qw < 0.0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        dot = qx*axis[0] + qy*axis[1] + qz*axis[2]
        return math.degrees(2.0 * math.atan2(dot, qw))

    def __wrap_angle_deg(self, angle_deg):
        return (angle_deg + 180.0) % 360.0 - 180.0

    def __quaternion_angle_deg(self, q1, q2):
        q1 = self.__normalize_quaternion(q1)
        q2 = self.__normalize_quaternion(q2)
        dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(2.0 * math.acos(dot))

    def __evaluate_accuracy(self, cp, pos, quat):
        dist = self.__dist(pos, cp['position'])
        marker_pos  = cp.get('marker_position')
        target_quat = cp.get('quaternion')
        centering   = None
        roll        = None
        if marker_pos is not None and target_quat is not None:
            view_axis = self.__normalize_vector([
                float(marker_pos[i]) - float(pos[i]) for i in range(3)
            ])
            cam_fwd = self.__normalize_vector(
                self.__rotate_vector_by_quaternion([1.0, 0.0, 0.0], quat)
            )
            centering = self.__vector_angle_deg(cam_fwd, view_axis)
            roll_actual = self.__twist_angle_deg(quat, view_axis)
            roll_target = self.__twist_angle_deg(target_quat, view_axis)
            roll = abs(self.__wrap_angle_deg(roll_actual - roll_target))
        return dist, centering, roll

    def __within(self, cp, pos, quat, max_d, max_a):
        d, c, r = self.__evaluate_accuracy(cp, pos, quat)
        if d > max_d:
            return False, d, c, r
        if c is not None and c > max_a:
            return False, d, c, r
        if r is not None and r > max_a:
            return False, d, c, r
        return True, d, c, r

    # ==================================================================
    # ArUco Detection (Background Async)
    # ==================================================================

    def __scan_aruco_at_checkpoint(self, cp_id):
        try:
            rospy.sleep(self.ARUCO_CAPTURE_WAIT_SEC)
            with self.__cam_lock:
                if self.__latest_image is None or self.__camera_matrix is None:
                    return
                img = self.__latest_image.copy()
            corners, ids, _ = aruco.detectMarkers(
                img, self.__aruco_dict, parameters=self.__aruco_params)
            if ids is not None and len(ids) > 0:
                with self.__aruco_lock:
                    for marker_id in ids.flatten():
                        self.__aruco_seen[int(marker_id)] = cp_id
                rospy.loginfo('[ARUCO] CP-%02d detected marker IDs: %s',
                              cp_id, ids.flatten().tolist())
        except Exception as e:
            rospy.logwarn('[ARUCO] scan error: %s', e)
        finally:
            with self.__aruco_worker_lock:
                self.__aruco_worker_running = False

    def __maybe_scan_aruco(self, cp_id):
        if not self.ENABLE_ARUCO_SCAN or aruco is None:
            return
        with self.__aruco_worker_lock:
            if self.__aruco_worker_running:
                return
            self.__aruco_worker_running = True
        worker = threading.Thread(target=self.__scan_aruco_at_checkpoint, args=(cp_id,))
        worker.daemon = True
        worker.start()

    # ==================================================================
    # Event helpers
    # ==================================================================

    def __notify_mission_start(self):
        rospy.wait_for_service('/event/mission_start')
        res = self.__mission_start()
        rospy.loginfo('[MISSION] Start: success=%s', res.success)
        return res.success

    def __notify_mission_finished(self):
        rospy.wait_for_service('/event/mission_finished')
        res = self.__mission_finished()
        rospy.loginfo('[MISSION] Finished: success=%s', res.success)
        return res.success

    def __notify_checkpoint_passed(self, checkpoint_id):
        rospy.loginfo('[CP] Notifying CP-%02d...', checkpoint_id)
        self.__checkpoint_client.wait_for_server()
        goal = CheckpointPassedGoal()
        goal.checkpoint_id = checkpoint_id
        self.__checkpoint_client.send_goal(goal)
        while not rospy.is_shutdown():
            if self.__checkpoint_client.wait_for_result(rospy.Duration(0.05)):
                result = self.__checkpoint_client.get_result()
                rospy.loginfo('[CP] CP-%02d result: success=%s', checkpoint_id, result.success)
                return result.success
        if rospy.is_shutdown():
            self.__checkpoint_client.cancel_goal()
        return False

    # ==================================================================
    # Checkpoint Data & Route Planning (Held-Karp TSP)
    # ==================================================================

    def __get_checkpoint_id(self, checkpoint):
        for key in ('checkpoint_id', 'checkpointid', 'id'):
            if key in checkpoint:
                return int(checkpoint[key])
        return None

    def __resolve_all_checkpoints(self):
        rospy.wait_for_service('/mission/get_checkpoints')
        res = self.__checkpoints()
        if not res.success:
            rospy.logerr('[CP] get_checkpoints failed: %s', res.message)
            return [], None
        try:
            payload      = json.loads(res.message)
            raw_cps      = payload.get('checkpoints', payload.get('Checkpoints', []))
            goalpoint_id = payload.get('goalpoint_id', None)
            if goalpoint_id is not None:
                goalpoint_id = int(goalpoint_id)
            checkpoints = []
            for cp in raw_cps:
                if not isinstance(cp, dict):
                    continue
                cp_id = self.__get_checkpoint_id(cp)
                if cp_id is None:
                    continue
                pos = cp.get('position', [0.0, 0.0, 0.0])
                if not (isinstance(pos, list) and len(pos) >= 3):
                    continue
                quat = [0.0, 0.0, 0.0, 1.0]
                att_obj = cp.get('attitude')
                if isinstance(att_obj, dict):
                    aq = att_obj.get('quaternion')
                    if isinstance(aq, list) and len(aq) >= 4:
                        quat = [float(v) for v in aq[:4]]
                marker_obj = cp.get('marker_pose')
                marker_pos = None
                if isinstance(marker_obj, dict):
                    mp = marker_obj.get('position')
                    if isinstance(mp, list) and len(mp) >= 3:
                        marker_pos = [float(mp[0]), float(mp[1]), float(mp[2])]
                checkpoints.append({
                    'checkpoint_id':   cp_id,
                    'position':        [float(pos[0]), float(pos[1]), float(pos[2])],
                    'quaternion':      quat,
                    'marker_position': marker_pos,
                })
            rospy.loginfo('[CP] Parsed %d checkpoints. Goalpoint ID: %s',
                          len(checkpoints), goalpoint_id)
            return checkpoints, goalpoint_id
        except Exception as exc:
            rospy.logerr('[CP] Failed to parse payload: %s', exc)
            return [], None

    def __leg_time(self, p1, q1, p2, q2):
        dist = self.__dist(p1, p2)
        turn = self.__quaternion_angle_deg(q1, q2)
        t_trans = dist / self.MOVE_VMAX_MPS
        t_rot   = turn / self.ROT_WMAX_DPS
        # Built-in controller mengeksekusi translasi lalu rotasi (POS_ATT)
        return t_trans + t_rot + 0.30

    def __plan_route(self, checkpoints, start_pos, start_quat, goal_pos=None, goal_quat=None):
        count = len(checkpoints)
        if count <= 1:
            return list(checkpoints)

        pts   = [cp['position']   for cp in checkpoints]
        quats = [cp['quaternion'] for cp in checkpoints]

        def goal_cost(idx):
            if goal_pos is None or goal_quat is None:
                return 0.0
            return self.__leg_time(pts[idx], quats[idx], goal_pos, goal_quat)

        if count > 12:
            # Greedy jika terlalu banyak
            unvisited = list(range(count))
            order = []
            cur_p, cur_q = start_pos, start_quat
            while unvisited:
                best_j, best_c = None, float('inf')
                for j in unvisited:
                    c = self.__leg_time(cur_p, cur_q, pts[j], quats[j])
                    if c < best_c:
                        best_c, best_j = c, j
                order.append(best_j)
                unvisited.remove(best_j)
                cur_p, cur_q = pts[best_j], quats[best_j]
            return [checkpoints[i] for i in order]

        d = [[self.__leg_time(pts[i], quats[i], pts[j], quats[j]) for j in range(count)]
             for i in range(count)]
        d_start = [self.__leg_time(start_pos, start_quat, pts[i], quats[i]) for i in range(count)]
        d_goal  = [goal_cost(i) for i in range(count)]

        inf = float('inf')
        dp_cost   = [inf] * ((1 << count) * count)
        dp_parent = [-1]  * ((1 << count) * count)

        for i in range(count):
            dp_cost[(1 << i) * count + i] = d_start[i]

        for mask in range(1, 1 << count):
            base_mask = mask * count
            for last in range(count):
                cost = dp_cost[base_mask + last]
                if cost == inf:
                    continue
                for nxt in range(count):
                    if not (mask & (1 << nxt)):
                        nxt_idx = (mask | (1 << nxt)) * count + nxt
                        cand = cost + d[last][nxt]
                        if cand < dp_cost[nxt_idx]:
                            dp_cost[nxt_idx] = cand
                            dp_parent[nxt_idx] = last

        full_mask = ((1 << count) - 1) * count
        best_cost, best_last = inf, -1
        for i in range(count):
            total_cand = dp_cost[full_mask + i] + d_goal[i]
            if total_cand < best_cost:
                best_cost, best_last = total_cand, i

        order, curr, m = [], best_last, (1 << count) - 1
        while curr != -1:
            order.append(curr)
            prev = dp_parent[m * count + curr]
            m ^= (1 << curr)
            curr = prev
        order.reverse()

        route = [checkpoints[idx] for idx in order]
        rospy.loginfo('[ROUTE] Held-Karp route (est %.1fs): %s -> Goal',
                      best_cost, [cp['checkpoint_id'] for cp in route])
        return route

    def __compute_timeout(self, cp, current_pos, current_quat):
        dist  = self.__dist(current_pos, cp['position'])
        turn  = self.__quaternion_angle_deg(current_quat, cp['quaternion'])
        t_est = self.__leg_time(current_pos, current_quat, cp['position'], cp['quaternion'])
        timeout = max(14.0, min(26.0, 5.0 + 1.2 * t_est))
        elapsed   = self.__mission_elapsed()
        remaining = self.TIME_LIMIT_SEC - elapsed - self.TIME_SAFETY_SEC
        timeout   = min(timeout, max(8.0, remaining * 0.50))
        rospy.loginfo('[TIMEOUT] CP-%02d dist=%.2fm turn=%.0fdeg est=%.1fs timeout=%.1fs',
                      cp['checkpoint_id'], dist, turn, t_est, timeout)
        return timeout

    # ==================================================================
    # Motion Execution & Handshake
    # ==================================================================

    def __send_pose_goal(self, position, quaternion):
        target = PoseStamped()
        target.header.frame_id    = 'iss_body'
        target.header.stamp       = rospy.Time.now()
        target.pose.position.x    = position[0]
        target.pose.position.y    = position[1]
        target.pose.position.z    = position[2]
        target.pose.orientation.x = quaternion[0]
        target.pose.orientation.y = quaternion[1]
        target.pose.orientation.z = quaternion[2]
        target.pose.orientation.w = quaternion[3]
        command_type = CtlStatusType(type=CtlStatusType.MOVE_TO_ABSOLUTE_TARGET)
        goal = CtlCommandGoal(target=target, type=command_type)
        self.__action_active = True
        self.__command.send_goal(goal)

    def __release_goal(self, wait_sec=0.0):
        try:
            if self.__action_active:
                self.__command.cancel_goal()
                if wait_sec > 0.0:
                    self.__command.wait_for_result(rospy.Duration(wait_sec))
        except Exception:
            pass
        self.__action_active = False

    def __after_motion_result(self, cp, motion_ok, is_goal=False):
        """Dipanggil jika motion action selesai sebelum in-flight handshake."""
        t_wait = rospy.Time.now()
        last_seq = -1
        while not rospy.is_shutdown() and (rospy.Time.now()-t_wait).to_sec() < self.AFTER_MOTION_WAIT_SEC:
            nav = self.__get_nav()
            if nav is not None and nav[3] != last_seq:
                last_seq = nav[3]
                ok, d, c, r = self.__within(cp, nav[0], nav[1], self.FAST_PASS_DIST, self.FAST_PASS_ANGLE)
                if ok:
                    rospy.loginfo('[NAV] CP-%02d Lv3 confirmed: d=%.3f c=%.1f r=%.1f',
                                  cp['checkpoint_id'], d, c or 0.0, r or 0.0)
                    return self.__notify_checkpoint_passed(cp['checkpoint_id'])
            rospy.sleep(0.02)
        # Fallback Lv2
        nav = self.__get_nav()
        if nav is not None:
            ok, d, c, r = self.__within(cp, nav[0], nav[1], self.LENIENT_DIST, self.LENIENT_ANGLE)
            if ok:
                rospy.logwarn('[NAV] CP-%02d fallback Lv2 (d=%.3f)', cp['checkpoint_id'], d)
                return self.__notify_checkpoint_passed(cp['checkpoint_id'])
        return None

    def __after_timeout(self, cp):
        nav = self.__get_nav()
        if nav is not None:
            ok, d, c, r = self.__within(cp, nav[0], nav[1], self.LENIENT_DIST, self.LENIENT_ANGLE)
            if ok:
                rospy.logwarn('[NAV] CP-%02d timeout tapi dalam Lv2 (d=%.3f)', cp['checkpoint_id'], d)
                return self.__notify_checkpoint_passed(cp['checkpoint_id'])
        return None

    def __visit_checkpoint(self, cp, timeout_sec, is_goal=False):
        """
        Terbang ke CP menggunakan built-in controller dan lapor sedini mungkin.
        """
        cp_id = cp['checkpoint_id']
        t0 = rospy.Time.now()
        try:
            self.__send_pose_goal(cp['position'], cp['quaternion'])
            t_final      = rospy.Time.now()
            stable       = 0
            within_since = None
            last_seq     = -1

            rospy.sleep(0.06)

            while not rospy.is_shutdown():
                now     = rospy.Time.now()
                elapsed = (now - t0).to_sec()
                if elapsed > timeout_sec:
                    rospy.logwarn('[NAV] CP-%02d timeout (%.1fs).', cp_id, timeout_sec)
                    self.__release_goal(wait_sec=0.1)
                    return self.__after_timeout(cp)

                nav = self.__get_nav()
                if nav is not None and nav[3] != last_seq:
                    last_seq = nav[3]
                    ok, d, c, r = self.__within(cp, nav[0], nav[1],
                                                self.FAST_PASS_DIST, self.FAST_PASS_ANGLE)
                    if ok:
                        stable += 1
                        if within_since is None:
                            within_since = now
                    else:
                        stable = 0
                        within_since = None

                    held_sec = 0.0 if within_since is None else (now - within_since).to_sec()
                    if stable >= self.REQUIRED_STABLE_SAMPLES and held_sec >= self.VALIDATION_HOLD_SEC:
                        rospy.loginfo(
                            '[NAV] In-flight Handshake CP-%02d (Lv3): d=%.3f c=%.1f r=%.1f t=%.1fs',
                            cp_id, d, c or 0.0, r or 0.0, elapsed)
                        return self.__notify_checkpoint_passed(cp_id)

                if (now - t_final).to_sec() > 0.3:
                    if self.__command.wait_for_result(rospy.Duration(0.02)):
                        result = self.__command.get_result()
                        motion_ok = (result is not None and
                                     result.type == CtlCommandResult.TERMINATE_SUCCESS)
                        return self.__after_motion_result(cp, motion_ok, is_goal)

                rospy.sleep(0.02)
            return None
        except Exception as e:
            rospy.logerr('[NAV] visit CP-%02d error: %s', cp_id, e)
            self.__release_goal(wait_sec=0.1)
            return None

    # ==================================================================
    # Main Flow
    # ==================================================================

    def __mission_elapsed(self):
        if self.__t_start is None:
            return 0.0
        return (rospy.Time.now() - self.__t_start).to_sec()

    def __tune_controller(self):
        """Naikkan kecepatan dan gaya robot via parameter ROS dalam batas hardware legal."""
        try:
            rospy.set_param('/ctl/tolerance_att', 0.05)
            rospy.set_param('/ctl/duration_goal', 0.20)   # dwell dipersingkat dari 3.0s ke 0.20s
            rospy.set_param('/pos_profile/v_max', 0.25)   # batas maksimum RangeChecker <= 0.25 m/s
            rospy.set_param('/pos_profile/f_max', 0.48)   # batas maksimum RangeChecker <= 0.50 N
            rospy.set_param('/att_profile/w_max', 0.5236) # batas maksimum RangeChecker <= 0.5236 rad/s
            rospy.set_param('/att_profile/t_max', 0.019)  # batas maksimum RangeChecker <= 0.020 Nm
            from ib2_msgs.srv import UpdateParameter
            rospy.wait_for_service('/ctl/update_params', timeout=0.5)
            update_srv = rospy.ServiceProxy('/ctl/update_params', UpdateParameter)
            response   = update_srv()
            success_code = getattr(UpdateParameter._response_class, 'SUCCESS', 0)
            if getattr(response, 'status', None) == success_code:
                self.__controller_tuned = True
                rospy.loginfo('[CTL] Tuned: v=0.25 F=0.48 w=0.5236 T=0.019 dwell=0.20s')
            else:
                rospy.logwarn('[CTL] update_params status=%s',
                              getattr(response, 'status', 'unknown'))
        except Exception as ctl_err:
            rospy.logwarn('[CTL] update_params note: %s', ctl_err)

    def __run_user_flow(self, msg):
        try:
            self.__tune_controller()

            if not self.__notify_mission_start():
                rospy.logerr('Failed to start mission')
                return

            checkpoints, goalpoint_id = self.__resolve_all_checkpoints()
            if not checkpoints:
                rospy.logerr('No checkpoints received, aborting')
                return

            intermediate_cps = [c for c in checkpoints if c['checkpoint_id'] != goalpoint_id]
            goal_cp   = next((c for c in checkpoints if c['checkpoint_id'] == goalpoint_id), None)
            goal_pos  = goal_cp['position']  if goal_cp else None
            goal_quat = goal_cp['quaternion'] if goal_cp else None

            # Tunggu nav data
            t_nav_wait = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now()-t_nav_wait).to_sec() < 1.0:
                if self.__get_nav() is not None:
                    break
                rospy.sleep(0.05)

            start_pos  = [0.20, 0.00, 0.00]
            start_quat = [0.0, 0.0, 0.0, 1.0]
            nav = self.__get_nav()
            if nav is not None:
                start_pos, start_quat = nav[0], nav[1]

            ordered_cps = self.__plan_route(
                intermediate_cps, start_pos=start_pos, start_quat=start_quat,
                goal_pos=goal_pos, goal_quat=goal_quat)

            rospy.loginfo('=== [PLAN] %d CP. Rute: %s -> Goal CP-%s ===',
                          len(checkpoints), [c['checkpoint_id'] for c in ordered_cps], goalpoint_id)

            passed_count = 0

            for cp in ordered_cps:
                if rospy.is_shutdown():
                    break
                cp_id   = cp['checkpoint_id']
                elapsed = self.__mission_elapsed()

                if elapsed >= self.TIME_LIMIT_SEC - self.TIME_SAFETY_SEC:
                    rospy.logwarn('>>> CP-%02d DILEWATI (waktu kritis t=%.0fs)', cp_id, elapsed)
                    continue

                nav_now = self.__get_nav()
                if nav_now is not None:
                    timeout = self.__compute_timeout(cp, nav_now[0], nav_now[1])
                else:
                    timeout = self.MOVE_TIMEOUT_BASE + 15.0

                rospy.loginfo('>>> CP-%02d (t=%.0fs, timeout=%.0fs)', cp_id, elapsed, timeout)
                t_visit = rospy.Time.now()
                res = self.__visit_checkpoint(cp, timeout, is_goal=False)
                if res is True:
                    passed_count += 1
                    self.__maybe_scan_aruco(cp_id)
                rospy.loginfo('[TIMING] CP-%02d visit=%.1fs mission=%.1fs result=%s',
                              cp_id, (rospy.Time.now()-t_visit).to_sec(),
                              self.__mission_elapsed(), res)

            # Capai Goal Checkpoint
            if goal_cp is not None and not rospy.is_shutdown():
                elapsed = self.__mission_elapsed()
                if elapsed < self.TIME_LIMIT_SEC - self.TIME_SAFETY_SEC:
                    nav_now = self.__get_nav()
                    if nav_now is not None:
                        timeout = min(self.__compute_timeout(goal_cp, nav_now[0], nav_now[1]),
                                      self.GOAL_TIMEOUT_SEC)
                    else:
                        timeout = max(10.0, min(self.GOAL_TIMEOUT_SEC,
                                                self.TIME_LIMIT_SEC - 3.0 - elapsed))
                    rospy.loginfo('>>> GOAL CP-%02d (t=%.0fs, timeout=%.0fs)',
                                  goalpoint_id, elapsed, timeout)
                    t_visit = rospy.Time.now()
                    res = self.__visit_checkpoint(goal_cp, timeout, is_goal=True)
                    if res is True:
                        passed_count += 1
                        self.__maybe_scan_aruco(goalpoint_id)
                    rospy.loginfo('[TIMING] GOAL CP-%02d visit=%.1fs mission=%.1fs result=%s',
                                  goalpoint_id, (rospy.Time.now()-t_visit).to_sec(),
                                  self.__mission_elapsed(), res)
                else:
                    rospy.logwarn('>>> GOAL DILEWATI (waktu kritis t=%.0fs)', elapsed)

            if self.ENABLE_ARUCO_SCAN:
                with self.__aruco_worker_lock:
                    scan_active = self.__aruco_worker_running
                if scan_active:
                    rospy.sleep(self.ARUCO_CAPTURE_WAIT_SEC)
                with self.__aruco_lock:
                    captured = sorted(self.__aruco_seen.keys())
                rospy.loginfo('[ARUCO] captured IDs: %s', captured)

            rospy.loginfo('=== [HASIL] %d/%d CP diterima, waktu %.1fs ===',
                          passed_count, len(checkpoints), self.__mission_elapsed())

            if not rospy.is_shutdown():
                self.__notify_mission_finished()
            self.__release_goal()

        except Exception as exc:
            rospy.logerr('user flow error: %s', exc)
            self.__release_goal()

    # ==================================================================
    # ROS Callbacks
    # ==================================================================

    def __callback_start(self, msg):
        if self.__started:
            rospy.logwarn('start callback already processed')
            return
        self.__started = True
        self.__t_start = rospy.Time.now()
        rospy.loginfo('Mission start signal received: %s', msg)
        self.__run_user_flow(msg)

    def __stop_processing(self, request):
        try:
            if self.__action_active:
                self.__command.cancel_goal()
            rospy.loginfo('Finish processing stop request')
            return StopProcessingUserNodeResponse(result=StopProcessingUserNodeResponse.SUCCESS)
        except Exception as exc:
            rospy.logerr('Stop processing failed: %s', exc)
            return StopProcessingUserNodeResponse(result=StopProcessingUserNodeResponse.ERROR)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    UserTemplate().run()
