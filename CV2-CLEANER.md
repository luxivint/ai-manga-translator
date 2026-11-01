# CV2-only çeviri worker sürümü

## Aktif akış

API detection → mevcut OCR/çeviri API → ilk segmentasyon maskesi → uygun serbest metinde CV2 dar kontur maskesi → CV2 dolgu → mevcut render / yayın doğrulaması.

LaMa, hibrit motor, ikinci segmentasyon/residual/QA onarım döngüsü ve bunlara ait model indirme kaydı kaldırıldı. Eski motor seçimi iş alınmadan önce açık hata üretir. İlk maskeyi çıkaran YOLO segmenter ayrı bir görevdir; PyTorch/ultralytics onun için hâlâ gereklidir. CV2 doldurma model/GPU kullanmaz.

Dar maske yalnızca yeterli karşıtlık kanıtı olan `text_free` bloklarda ve mevcut maskenin geniş olduğu durumlarda uygulanır. Konturu kapsar, mevcut maskenin dışına çıkmaz. Balon, reklam, belirsiz veya zaten dar maskede önceki maske korunur. Denemelerde başarılı olan her örneğin bütün serilerde kusursuz olacağı iddia edilmez.

## Ortak PC / sunucu ayarı

```dotenv
CLEAN_ENGINE=cv2
CLEAN_CV2_MODE=adaptive
CLEAN_CV2_RADIUS=3
CLEAN_CV2_REFINE_MASK=true
CLEAN_POSTPROCESS=false
OPENCV_NUM_THREADS=2
BATCH_WORKERS=1
BATCH_WORKERS_AUTO=false
BATCH_WORKERS_MAX=1
DETECTION_BACKEND=api
```

`adaptive`: uygun düz/renk geçişli zeminde renk düzlemi, diğer yerde Telea 3. `ns` açık seçenek olarak desteklenir. Deneysel patch/coarse seçenekleri üretimden çıkarıldı. Detection API seçildiğinde yerel RT-DETR geri dönüşü yoktur; API başarısızsa ham görseller korunur.

PC .env ve TrendTray translation profili bu ayarlarla güncellendi; diğer servisler korunur. Çalışan bir iş zorla kesilmez. Worker başlatılınca `Cleaner ready` satırından CV2 profili doğrulanır. Sürecin açık olması, gerçek bir iş tamamlandığı anlamına gelmez.

## Docker / Dokploy

- Dockerfile CPU-only torch/ONNX doğrulaması, OpenCV 4.12 / NumPy 2.2.6, gerçek CV2 dolgu + Türkçe font smoke testi içerir.
- Ultralytics'in getirdiği normal OpenCV payload'ı CPU imajında kaldırılıp aynı sürüm headless payload geri kurulur. Model dosyaları, .env, yerel çıktılar ve araştırma araçları build bağlamından hariç tutulur.
- `compose.cpu.yml` CV2 ve API detection profilini açıkça geçirir. SIGTERM aktif işi tamamlayıp durmaya devam eder; work volume ve sırlar korunur.
- Dokploy mevcut uygulamasında build path `/`, Dockerfile `Dockerfile` olarak kalır. Environment içinde yukarıdaki profil, mevcut gerçek `DATABASE_URL`, R2 ve API bağlantı değerleri bulunmalıdır. Eski `LAMA_*`, `CLEAN_LAMA_*` ve hibrit/QA/residual ayarlarını kaldırın.
- Sunucuda ilk maske modelinin indirilmesi gerekebilir; cache `/app/work` volume altında kalır. Bu süre dolgu benchmarkına dahil değildir.

Mevcut secret dosyasını koruyarak standalone kurulum:

```sh
docker compose --env-file .env.server -f compose.cpu.yml build
docker compose --env-file .env.server -f compose.cpu.yml up -d
docker compose --env-file .env.server -f compose.cpu.yml logs -f translation-worker
```

Aktif iş varsa deploy öncesinde bitmesini bekleyin; kuyruk/publish doğrulama davranışı korunmuştur. Dokploy'da aynı commit'in `Rebuild` ile yeniden derlenmesi gerekir; yalnızca reload eski imajı değiştirmez.

## Doğrulama sınırı

Yerel birim/regresyon sonuçları ve hazır maskeli 113 sayfa testi `Manga/artifacts/cv2-release/` altında tutulur. Gerçek API/DB/R2 yayını bu hazırlık testi sırasında yapılmaz. Bu PC'de Docker executable/daemon bulunmadığından Linux imajı burada derlenip çalıştırılmış sayılmaz; Dockerfile build aşamasında smoke kontrolü zorunludur. Canlı deploy ve push bu hazırlık kapsamında yapılmadı.
