import json
import pandas as pd

# 模型映射
label_to_model = {
    "A": "qwen3-30b",
    "B": "llama-4-maverick",
    "C": "claude-sonnet-4",
    "D": "gpt-4o-mini",
    "E": "Deepseek V3"
}

data_path = "D:/Deeplearning/Change_Agent_main/sampled_result/CoT_Result/LEVIR_Text_background/full_results.json"
# 加载数据
records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去除首尾空白字符
        if not line:  # 跳过空行
            continue
        try:
            data = json.loads(line)
            orig = data.get("Original Caption", "").lower()
            scores = data.get("Scores", {})
            change_type = "No Change" if any(x in orig for x in ["no change", "unchanged", "same as before"]) else "Changed"
            for label, score in scores.items():
                if label in label_to_model:
                    records.append({
                        "Original Caption": data["Original Caption"],
                        "Change Type": change_type,
                        "Model": label_to_model[label],
                        "Score": score
                    })
        except json.JSONDecodeError as e:
            print(f"跳过无法解析的行: {line}，错误: {e}")
            continue

df = pd.DataFrame(records)

# ✅ 表格 1：总体评分表
overall = df.groupby("Model").agg(
    Count=("Score", "count"),
    Mean_Score=("Score", "mean"),
    Std_Dev=("Score", "std")
).reset_index().sort_values(by="Mean_Score", ascending=False)

# ✅ 表格 2：按变化/未变化分类评分表
grouped = df.groupby(["Change Type", "Model"]).agg(
    Count=("Score", "count"),
    Mean_Score=("Score", "mean"),
    Std_Dev=("Score", "std")
).reset_index().sort_values(["Change Type", "Mean_Score"], ascending=[True, False])

# 保存为CSV（或用于复制到论文表格）
overall.to_csv("D:/Deeplearning/Change_Agent_main/sampled_result/CoT_Result/LEVIR_Text_background/overall_scores.csv", index=False)
grouped.to_csv("D:/Deeplearning/Change_Agent_main/sampled_result/CoT_Result/LEVIR_Text_background/scores_by_change_type.csv", index=False)

# 可选：打印结果
print("\n📊 总体模型评分:")
print(overall.round(2))
print("\n📊 按变化类型分类评分:")
print(grouped.round(2))
