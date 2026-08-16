#!/usr/bin/env python3
"""
模拟硬件服务器 —— 机械臂(8087) + 灵巧手(8088)
=============================================
用途: 端到端验证算法服务的控制程序正确性

1. 严格按官方 API 契约响应（success/字段名/错误行为）
2. 记录所有收到的指令（路径+请求体+时间）到 /tmp/fake_hardware_log.jsonl
3. 模拟运动过程（POST 后 moving=true，1s 后恢复）
4. 安全工作域校验（越界指令直接返回失败）—— 发现控制程序 bug 的关键
"""
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ARM_PORT = 8087
HAND_PORT = 8088
LOG_FILE = "/tmp/fake_hardware_log.jsonl"

# 安全工作域（与 config.py ARM_WORKSPACE 一致，单位 m）
WS_Y = (-0.28, -0.04)
WS_Z = (0.44, 0.52)

arm_state = {
    "enabled": False,
    "moving": False,
    "pose": {"x": 0.30, "y": -0.16, "z": 0.48, "roll": 3.14, "pitch": 0.0, "yaw": 0.0},
    "mode": "idle",
}
hand_state = {"connected": True, "position": [0.0] * 10, "errors": [0] * 10}

log_lock = threading.Lock()


def log_request(who: str, method: str, path: str, body: dict):
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": round(time.time(), 3),
                "who": who,
                "method": method,
                "path": path,
                "body": body,
            }, ensure_ascii=False) + "\n")


def ok(**kw):
    return {"success": True, "message": "ok", **kw}


def fail(msg):
    return {"success": False, "message": msg}


class ArmHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    def _send(self, data, code=200):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?")[0]
        log_request("arm", "GET", path, {})
        if path == "/api/status":
            self._send(ok(moving=arm_state["moving"], joints=[0.0] * 7))
        elif path == "/api/pose":
            self._send(ok(pose=arm_state["pose"]))
        else:
            self._send(fail(f"未知端点 {path}"), 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_body()
        log_request("arm", "POST", path, body)

        if path == "/api/enable":
            arm_state["enabled"] = True
            # 官方格式: {"right": {"success": true, "message": "..."}}（嵌套）
            self._send({"right": {"success": True, "message": "motors enabled"}})
        elif path == "/api/disable":
            arm_state["enabled"] = False
            self._send({"right": {"success": True, "message": "7 motors disabled"}})
        elif path == "/api/end_effector":
            # 安全工作域校验（关键：发现越界指令）
            right = body.get("right", {})
            y = right.get("y", 0)
            z = right.get("z", 0)
            if not (WS_Y[0] <= y <= WS_Y[1]):
                self._send(fail(f"Y={y} 超出安全工作域 {WS_Y}"))
                return
            if not (WS_Z[0] <= z <= WS_Z[1]):
                self._send(fail(f"Z={z} 超出安全工作域 {WS_Z}"))
                return
            if not arm_state["enabled"]:
                self._send(fail("电机未使能"))
                return
            # 模拟运动：1 秒后到位
            arm_state["moving"] = True
            arm_state["pose"].update(right)

            def _settle():
                time.sleep(1.0)
                arm_state["moving"] = False
            threading.Thread(target=_settle, daemon=True).start()
            self._send(ok(message="直线运动完成"))
        elif path == "/api/move_joints":
            if not arm_state["enabled"]:
                self._send(fail("电机未使能"))
                return
            joints = body.get("joints", [])
            if len(joints) != 7:
                self._send(fail(f"需要7个关节角，收到{len(joints)}"))
                return
            arm_state["moving"] = True

            def _settle():
                time.sleep(1.0)
                arm_state["moving"] = False
            threading.Thread(target=_settle, daemon=True).start()
            self._send(ok(message="关节运动完成"))
        elif path == "/api/move_home":
            arm_state["pose"] = {"x": 0.30, "y": -0.16, "z": 0.48,
                                 "roll": 3.14, "pitch": 0.0, "yaw": 0.0}
            self._send(ok(message="已回原点"))
        elif path.startswith("/api/teach") or path.startswith("/api/playback"):
            self._send(ok(message="示教/回放命令已受理"))
        else:
            self._send(fail(f"未知端点 {path}"), 404)


class HandHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    def _send(self, data, code=200):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?")[0]
        log_request("hand", "GET", path, {})
        if path == "/api/status":
            self._send(ok(connected=hand_state["connected"]))
        elif path == "/api/pose":
            self._send(ok(position=hand_state["position"]))
        elif path == "/api/errors":
            self._send(ok(error_codes=hand_state["errors"]))
        else:
            self._send(fail(f"未知端点 {path}"), 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_body()
        log_request("hand", "POST", path, body)

        if path == "/api/set_pos":
            pos = body.get("position", [])
            if len(pos) != 10:
                self._send(fail(f"需要10个位置值，收到{len(pos)}"))
                return
            if any(not (0.0 <= p <= 1.0) for p in pos):
                self._send(fail(f"位置值超出[0,1]: {pos}"))
                return
            hand_state["position"] = pos
            self._send(ok(message="位置设置成功"))
        else:
            self._send(fail(f"未知端点 {path}"), 404)


if __name__ == "__main__":
    # 清空日志
    open(LOG_FILE, "w").close()
    arm_srv = HTTPServer(("127.0.0.1", ARM_PORT), ArmHandler)
    hand_srv = HTTPServer(("127.0.0.1", HAND_PORT), HandHandler)
    print(f"模拟机械臂: http://127.0.0.1:{ARM_PORT}")
    print(f"模拟灵巧手: http://127.0.0.1:{HAND_PORT}")
    print(f"指令日志: {LOG_FILE}")
    threading.Thread(target=arm_srv.serve_forever, daemon=True).start()
    hand_srv.serve_forever()
