import os
import subprocess
import threading
import shutil

# meteor-1.5.jar 必须和这个文件在同一目录
METEOR_JAR = 'meteor-1.5.jar'


class Meteor:
    def __init__(self):
        # 找到绝对路径
        meteor_dir = os.path.dirname(os.path.abspath(__file__))
        meteor_jar_path = os.path.join(meteor_dir, METEOR_JAR)

        # 检查 meteor-1.5.jar 是否存在
        if not os.path.exists(meteor_jar_path):
            raise FileNotFoundError(f"找不到 meteor-1.5.jar 文件: {meteor_jar_path}")

        # 尝试找到 java 的完整路径
        java_path = r"C:\Program Files\Java\jre1.8.0_202\bin\java.exe"
        if java_path is None:
            raise EnvironmentError("找不到 Java 运行环境，请确认已经安装 Java 并配置到系统 PATH 中。")

        # 构建命令
        self.env = os.environ.copy()
        self.env['LC_ALL'] = 'en_US.UTF_8'

        self.meteor_cmd = [java_path, '-jar', '-Xmx2G', meteor_jar_path, '-', '-', '-stdio', '-l', 'en', '-norm']

        try:
            self.meteor_p = subprocess.Popen(
                self.meteor_cmd,
                cwd=meteor_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                universal_newlines=True,
                bufsize=1
            )
        except Exception as e:
            raise RuntimeError(f"启动 Meteor 失败，命令: {' '.join(self.meteor_cmd)}\n错误信息: {e}")

        # 初始化锁
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        scores = []
        eval_line = 'EVAL'
        self.lock.acquire()
        for i in range(len(res)):
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')
        self.meteor_p.stdin.flush()

        # Collect segment scores
        for i in range(len(res)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line + '\n')
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        if hasattr(self, 'lock'):
            self.lock.acquire()
            try:
                if hasattr(self, 'meteor_p'):
                    self.meteor_p.stdin.close()
                    self.meteor_p.kill()
                    self.meteor_p.wait()
            finally:
                self.lock.release()
