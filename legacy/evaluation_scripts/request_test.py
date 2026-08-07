import requests

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}

try:
    r = requests.get("https://api.siliconflow.cn", proxies=proxies, timeout=10)
    print("连接成功", r.status_code)
except Exception as e:
    print("连接失败:", e)