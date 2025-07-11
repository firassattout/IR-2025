نظام استرجاع المعلومات 2024-2025
نظرة عامة
هذا المشروع هو نظام استرجاع معلومات تم تطويره باستخدام Python لمعالجة مجموعتي بيانات: beir/quora/dev (522,931 وثيقة من الأسئلة والأجوبة القصيرة) وantique/test (403,665 وثيقة تحتوي على أسئلة وأجوبة عامية). يدعم النظام تمثيلات متعددة (TF-IDF، BERT، Hybrid، BM25)، فهرسة باستخدام Inverted Index، وميزة إضافية (Vector Stores باستخدام FAISS). يتميز النظام بواجهة ويب سهلة الاستخدام مبنية باستخدام FastAPI وJinja2.
بنية المشروع
يتبع النظام بنية الخدمات الموجهة (SOA) ويتكون من الخدمات التالية:

المعالجة المسبقة: تنظيف النصوص باستخدام Lemmatization وإزالة الكلمات الشائعة.
التمثيل: دعم TF-IDF، BERT، تمثيل هجين، وBM25.
الفهرسة: إنشاء فهارس عكسية باستخدام Whoosh.
معالجة الاستعلامات: تحويل الاستعلامات باستخدام نفس تقنيات التمثيل.
التقييم: حساب مقاييس الأداء (MAP، Recall، Precision@10، MRR).
واجهة المستخدم: واجهة ويب لإدخال الاستعلامات، اختيار النموذج، وعرض النتائج.
الميزة الإضافية: استخدام Vector Stores (FAISS) لتسريع البحث.

هيكلية الملفات

/data/: يحتوي على فهارس البيانات لـ beir/quora وantique/test.
/models/: يحتوي على النماذج المدربة (TF-IDF، BERT، Hybrid، BM25، FAISS).
/notebooks/: دفاتر Jupyter للمعالجة المسبقة، التمثيل، والتقييم.
/src/api/: واجهات برمجة التطبيقات (FastAPI) لكل خدمة.
/src/evaluation/: سكربتات لحساب مقاييس الأداء.
/src/indexing/: سكربتات بناء الفهارس.
/src/preprocessing/: سكربتات تنظيف البيانات.
/src/processing_ranking/: سكربتات معالجة الاستعلامات والترتيب.
/src/representation/: سكربتات التمثيل (TF-IDF، BERT، Hybrid، BM25).
/src/ui/: قوالب واجهة المستخدم (HTML).
/src/utils/: أدوات مساعدة مثل تنظيف النصوص.

متطلبات التشغيل

Python 3.8+
المكتبات: numpy, joblib, nltk, pymongo, fastapi, jinja2, whoosh, faiss-cpu, transformers, sentence-transformers
MongoDB (لتخزين البيانات)
تثبيت المتطلبات:pip install -r requirements.txt

التثبيت والتشغيل

استنساخ المستودع:git clone [رابط المستودع]
cd IR-2025

تثبيت المتطلبات:pip install -r requirements.txt

تشغيل خادم MongoDB.
تشغيل واجهة برمجة التطبيقات:uvicorn src.api.frontend_api:app --reload

الوصول إلى واجهة المستخدم عبر المتصفح: http://localhost:8000.

استخدام النظام

افتح واجهة المستخدم في المتصفح.
اختر مجموعة البيانات (beir/quora أو antique/test).
اختر النموذج (TF-IDF، BERT، Hybrid، FAISS TF-IDF، FAISS BERT، FAISS Hybrid).
أدخل الاستعلام وحدد عدد النتائج (k).
انقر على "Search" لعرض النتائج.

التقييم

beir/quora/dev:
FAISS BERT: MAP: 0.8493، Recall: 1.0، Precision@10: 0.0033، MRR: 0.8820
زمن التنفيذ: ~0.058 ثانية

antique/test:
FAISS Hybrid: MAP: 0.2052، Recall: 0.5680، Precision@10: 0.0364، MRR: 0.7669
زمن التنفيذ: ~0.053 ثانية

المساهمون

محمد فراس ستوت: المعالجة المسبقة، تصميم النظام.
محمد طرابلسي: التمثيلات، الفهرسة.
سارة دالاتي: واجهة المستخدم.
أحمد نعمة: التقييم، الميزة الإضافية.
