# Çeviri aşama ölçümleri

Ölçüm varsayılan olarak açıktır. PC worker ve Docker aynı kodu kullanır.
`PIPELINE_TRACE=false` kapatır; `PIPELINE_TRACE_HEARTBEAT_SECONDS=10` durum mesajı aralığıdır.
İstenirse `PIPELINE_TRACE_DIR` kayıt konumunu değiştirir. Varsayılan proje içindeki
`work/job-timings/` dizinidir; `/app/work` kalıcı diske bağlı olmalıdır.
Kayıtlar otomatik silinmez; iş başına JSONL olay kaydı ve JSON özet oluşur.
Dosyalar her deneme için benzersizdir; aynı iş yeniden denenirse önceki ölçümün üzerine yazılmaz.

## Görünen bilgiler

Dokploy Logs ve PC çeviri günlüğünde `[TIME] START/END` mesajları görünür.
Çocuk işlem mesajları denetleyen worker tarafından yaklaşık 5 saniyelik kontrollerde aktarılır.
`[TIME LIVE]` devam eden aşamaları, sayfayı, geçen süreyi, işlemci çekirdeği kullanımını
ve belleği gösterir. Bu durum mesajları takılma zaman aşımını sıfırlamaz.
Bir işlem tamamlanmadan worker öldürülürse JSON özet olmayabilir; JSONL'deki
başlangıç, bitiş ve örnekler yine incelenebilir.

Ölçülen ana işler:

- Worker hazırlığı, çocuk işlem beklemesi, veritabanına kayıt, yayın doğrulaması ve önbellek temizliği.
- R2 listeleme, indirme, yükleme, yükleme doğrulama/onarım ve silme.
- Recut düz alan taraması, detection pencereleri, kesim planlama, PNG kayıt ve piksel doğrulaması.
- Model hazırlığı, reklam taraması ve görsel çözme.
- Sayfa detection, font modeli/özellikleri, OCR, boş OCR kurtarma ve OCR tekrarları.
- Çeviri istekleri, çeviri değerlendirmesi ve yeniden deneme beklemesi.
- OCR'nin zaten bulduğu filigran kutularında yerel bölge maskesi ve düz renk dolgu.
  Bölüm çapında aday tarama, şablon öğrenme/eşleme kaldırıldı. Eski `WATERMARK_PLATES`
  ve `WATERMARK_PLATE_MINE` ayarları artık bu işlemleri etkinleştiremez. OCR'nin okumadığı
  filigranlar için ek tarama yapılmaz; bu işaretler görselde kalabilir.
- Segmenter hazırlığı/çıkarımı, maske çıkarma/iyileştirme, CV2 dolgu ve alan geri yükleme.
- Yazı yerleşimi, çizim, dosya kodlama/kayıt, boyut kontrolü ve damga.
- API kapasite sırası, devre kesici beklemesi, her HTTP girişimi, durum kodu ve yeniden deneme beklemesi.

HTTP süresi istemcide ölçülen istek-yanıt süresidir; sağlayıcının içindeki
sırada bekleme ile model hesaplamasını birbirinden ayıramaz. Recut ve detection
istekleri mevcut paralellik ve yeniden kullanım kurallarıyla aynen çalışır.
Ölçüm yeni model çağrısı yapmaz, kalite ayarlarını değiştirmez.

## Süreleri doğru okuma

`wall_s` geçen gerçek zamandır. `own_wall_s`, doğrudan alt işlemlerin kapsadığı
aralıkların birleşimi çıkarıldıktan sonra kalan zamandır. İç içe ve paralel
işlemlerin süreleri doğrudan toplanmaz. `active_wall_s` aynı aşamanın aktif
olduğu aralıkların birleşimidir; farklı aşamaların active süreleri de toplanmaz.
`own_work_wall_s` aynı aşamadaki işlerin toplamıdır, eşzamanlı sayfalarda toplam
geçen süreden fazla olabilir. Bir aşamanın own süresi yüksekse bu, o fonksiyon
içinde daha fazla ayrıntılandırılacak bir kısım olduğunu açıkça gösterir.

İşlemci zamanı saniye olarak kaydedilir. Örneğin bir saniyede 6 CPU saniyesi
`cpu_cores=6` demektir; 8 çekirdekli makinenin yaklaşık %75'ine karşılık gelir.
Çocuk ve ana worker ayrı süreçlerdir, ayrı raporlanır. Bir aşamanın gözlenen
process CPU değeri o sıradaki bütün süreç thread'lerini içerir; paralel aşamalar
arasında paylaştırılmış bir maliyet değildir. Bu değerler toplanmaz. Thread CPU
yerel modelin diğer thread'lerini kapsamaz. Bellek ve CPU tepe değerleri saniyelik
örneklerden gelir, çok kısa sıçramalar kaçabilir. GPU ölçümü yoktur; bu çalışma
CPU worker içindir.

## Raporu alma

Docker terminalinde:

```sh
python /app/scripts/job_timing_report.py job_KIMLIGI
python /app/scripts/job_timing_report.py job_KIMLIGI --bundle
```

İkinci komut yalnızca o işe ait logu, olayları ve özetleri
`/app/work/diagnostics/` altında tek `tar.gz` dosyasına toplar. Ortam/anahtar
dosyalarını arşive eklemez. Aşama kayıtları istek gövdesi, görsel, yanıt metni,
URL veya gizli anahtar içermez. Mevcut eski iş günlüğü kendi mevcut içeriğiyle
pakete girer. Dosyayı paylaşmadan önce gerektiğinde gözden geçir.

Eski işler yeni ölçüm dosyalarına sahip değildir; geçmişteki belirsiz süre bu
değişiklikle geriye dönük çözülemez. İlk yeni canlı işin raporu gerçek sunucu
dağılımını gösterecektir. Standart CLI'de ana modülün ilk import süresi ana worker'ın
çocuk bekleme süresine dahildir; sıcak runner'da ayrıca `startup.module_imports`
ölçülür. Kuyrukta bekleme çeviri süresine dahil değildir.
