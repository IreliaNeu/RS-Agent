import os
# test_meteor.py
from eval_func import Meteor

print(os.path.exists(r'C:\Program Files\Java\jre1.8.0_202\bin\java.exe'))

def test_meteor():
    m = Meteor()
    try:
        # 模拟计算分数
        gts = [['This is a reference sentence.']]
        res = [['This is a hypothesis sentence.']]
        score, _ = m.compute_score(gts, res)
        print(f"METEOR Score: {score}")
    finally:
        del m  # 显式触发析构

if __name__ == "__main__":
    test_meteor()