#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Kibo-RPC 7th Local Score Dashboard Viewer
Menampilkan hasil simulasi lokal persis seperti tampilan Web JAXA Dashboard.
"""
import os
import json
import glob

def find_latest_result():
    candidates = [
        "/home/henxyl/Downloads/Kibo RPC/krpc7th-simulator/share/logs/latest/judge_module_logs/result.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    logs = glob.glob("/home/henxyl/Downloads/Kibo RPC/krpc7th-simulator/share/logs/*/judge_module_logs/result.json")
    if logs:
        logs.sort(key=os.path.getmtime, reverse=True)
        return logs[0]
    return None

def main():
    result_path = find_latest_result()
    if not result_path or not os.path.exists(result_path):
        print("❌ File result.json tidak ditemukan. Jalankan simulation.sh terlebih dahulu.")
        return

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Gagal membaca result.json: {e}")
        return

    report = data.get("JudgeReport", {})
    mission = report.get("mission", {})
    checkpoints = report.get("checkpoint_detail", [])
    goal = report.get("goal", {})
    illegal = data.get("Illegal", False)

        # Hitung Skor Checkpoint
    # Level 3 = 10 pt, Level 2 = 5 pt, Level 1 = 2 pt
    level_scores = {3: 10.0, 2: 5.0, 1: 2.0, 0: 0.0}
    cp_score_total = 0.0
    passed_count = 0
    level_counts = {1: 0, 2: 0, 3: 0, "Failed": 0}

    table_rows = []
    for cp in checkpoints:
        cp_id = cp.get("checkpoint_id")
        success = cp.get("success", False)
        level = cp.get("level", 0)
        reason = cp.get("reason", "-")

        if success:
            passed_count += 1
            pts = level_scores.get(level, 10.0 if level == 3 else 0.0)
            cp_score_total += pts
            level_counts[level] = level_counts.get(level, 0) + 1
            status_str = "PASS"
            level_str = f"Lv{level}"
            score_str = f"{pts:.2f} pt"
        else:
            level_counts["Failed"] += 1
            status_str = "FAIL"
            level_str = "-"
            score_str = "0.00 pt"

        table_rows.append((f"CP-{cp_id:02d}", status_str, level_str, score_str, reason))

    # Goal status
    goal_success = goal.get("success", False)
    goal_reason = goal.get("reason", "-")
    goal_score = 50.0 if goal_success else 0.0
    # Durasi & Time Bonus
    duration = mission.get("duration_sec")
    if duration is not None:
        duration_str = f"{int(duration // 60):02d}:{duration % 60:04.1f}"
        # Formula time bonus JAXA Kibo RPC: max ~ 2.0 jika selesai cepat
        time_bonus = max(1.0, 2.0 - (duration / 600.0))
    else:
        duration_str = "Timeout / Unfinished"
        time_bonus = 1.0

    final_score = (cp_score_total + goal_score) * time_bonus if not illegal else 0.0

    print("=" * 68)
    print("           🚀 KIBO-RPC 7th - HASIL SIMULASI LOKAL 🚀")
    print("=" * 68)
    print(f"Status              : {'Finished' if mission.get('finished') else 'Incomplete / Stopped'}")
    print(f"Illegal Access      : {'YA (Pelanggaran)' if illegal else 'Tidak (Valid)'}")
    print(f"Mission Time        : {duration_str}")
    print(f"Time Bonus          : x {time_bonus:.2f}")
    print(f"CheckPoints Passed  : {passed_count} / {len(checkpoints)}")
    print(f"Goal Reached        : {'SUCCESS (PASS)' if goal_success else 'FAILED'} ({goal_reason})")
    print("-" * 68)
    print("📊 SCORE BREAKDOWN:")
    print(f"  • CheckPoint Score Total : {cp_score_total:.2f} pt")
    print(f"  • Goal Score             : {goal_score:.2f} pt")
    print(f"  • Level Summary          : Lv3: {level_counts[3]} | Lv2: {level_counts[2]} | Lv1: {level_counts[1]} | Fail: {level_counts['Failed']}")
    print(f"  ⭐ FINAL SCORE          : {final_score:.2f} pt")
    print("=" * 68)
    print(f"{'Check Point':<13} {'Result':<10} {'Level':<8} {'Score':<12} {'Reason'}")
    print("-" * 68)
    for cp_name, res, lvl, sc, rsn in table_rows:
        color_mark = "✅" if res == "PASS" else "❌"
        print(f"{color_mark} {cp_name:<10} {res:<10} {lvl:<8} {sc:<12} {rsn}")
    print("=" * 68)

if __name__ == "__main__":
    main()
