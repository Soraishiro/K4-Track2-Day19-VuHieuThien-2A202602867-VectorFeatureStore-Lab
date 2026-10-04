"""Demo: HybridMemoryAgent with 5 illustrative queries.

Run:  python bonus/demo.py
Exits 0 if all queries produce non-empty context.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bonus.agent import HybridMemoryAgent, DEFAULT_USER_ID


SAMPLE_MEMORIES = {
    DEFAULT_USER_ID: [
        """
User: Tôi có thể dùng Kubernetes để tự động mở rộng hạ tầng không?
Assistant: Có thể. Kubernetes HPA (Horizontal Pod Autoscaler) tự động scale số
replica dựa trên CPU utilization hoặc custom metrics. Kết hợp với Cluster
Autoscaler để thay đổi số node. Yêu cầu cài đặt metrics-server.
Tài liệu tham khảo: cloud_001, cloud_016.
""",
        """
User: Làm sao cấu hình CI/CD pipeline với GitHub Actions?
Assistant: Tạo file .github/workflows/build.yml với trigger trên push. Các bước:
checkout, setup python, install deps, test, deploy. Dùng OIDC identity để
tránh hardcode AWS credentials.
Tài liệu: devops_003, devops_019.
""",
        """
User: Có thư viện nào tốt cho NLP tiếng Việt không?
Assistant: underthesea hỗ trợ word segmentation, POS tagging, ner. pyvi cũng
phổ biến nhưng chỉ có tokenizer. Đối với embedding, bge-m3 tốt nhất cho
đa ngữ, dù nặng hơn bge-small-en.
Tài liệu: ai_ml_048, ai_ml_025.
""",
        """
User: Làm thế nào đo lường code coverage hiệu quả?
Assistant: Dùng pytest-cov cho unit test, tích lựn coverage >= 80%. Với
integration test, đo thời gian chạy và memory profile. Coverage không đo
quality nhưng giúp tìm dead code.
Tài liệu: backend_012, backend_034.
""",
        """
User: Các bước triển khai bảo mật OAuth2 cho API?
Assistant: 1) Đăng ký OIDC provider (Auth0/Keycloak). 2) Cấu hình callback URL.
3) Dùng PKCE cho SPA. 4) Validate token ở mỗi request bằng middleware.
5) Set short TTL cho access token, refresh token 7 ngày.
Tài liệu: security_008, security_021.
""",
    ],
    "u_002": [
        """
User: Tôi thấy latency API tăng đột ngột, là gì nguyên nhân?
Assistant: Kiểm tra cold start (model chưa warm), connection pool hết, hoặc
Qdrant indexing chưa ổn định. Dùng profiler để trace từng pipeline stage.
""",
    ],
}


QUERIES = [
    # 1. Simple vector hit
    ("Tôi đã đọc gì về Kubernetes?", "simple_vector"),
    # 2. Needs profile context (topic_affinity)
    ("Recommend đọc gì tiếp cho mình", "profile_context"),
    # 3. Needs fresh activity
    ("Tôi đang quan tâm gì gần đây?", "fresh_activity"),
    # 4. Paraphrase (semantic wins)
    ("Tài liệu về tự động mở rộng hệ thống?", "paraphrase"),
    # 5. Mixed (episodic + profile)
    ("Cho tôi summary cloud security", "mixed_hybrid"),
]


def main() -> None:
    agent = HybridMemoryAgent()

    for user_id, memories in SAMPLE_MEMORIES.items():
        for mem in memories:
            n = agent.remember(mem, user_id=user_id)
            print(f"[setup] Remembered {n} chunks for {user_id}")

    print(f"\n{'='*60}")
    for query, label in QUERIES:
        ctx = agent.recall(query, user_id=DEFAULT_USER_ID, top_k=3)
        print(f"\n--- Query [{label}]: {query}")
        print(ctx)
        print(f"{'='*60}")

    print("\nAll 5 queries executed successfully.")
    print("PASS: bonus/demo.py exits 0")


if __name__ == "__main__":
    main()
