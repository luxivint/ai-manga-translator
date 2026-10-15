# GPU olmayan Linux sunucuda çeviri worker’ı

Başlangıç: **1 worker, en fazla 2 vCPU / 8 GB RAM**. GPU veya NVIDIA sürücüsü
istenmez. PostgreSQL ve R2 mevcut sistemle aynıdır; Docker bu hizmetleri yeniden
oluşturmaz. 9router bu imajın içinde değildir ve sunucudan erişilebilir olmalıdır.
Bilgisayar kapalıyken çalışmak için 9router’ı da sunucuda veya erişilebilir başka
bir makinede çalıştırın. Worker dışarıya port açmaz.

## Hazırlık

Bu dosyalar **ai-manga-translator** deposundadır. Docker build bağlamı ve Compose
komutları bu deponun kök dizininde çalışır; Trendmanga üst deposunda değildir.

```sh
cp .env.server.example .env.server
```

`.env.server` dosyasında şunları doldurun:

| Değişken | Değer |
| --- | --- |
| `DATABASE_URL` | Mevcut Trendmanga PostgreSQL bağlantısı. Container’ın erişebildiği adres. |
| `R2_BUCKET` | Mevcut manga bucket adı. |
| `R2_ACCOUNT_ID` | Cloudflare hesap kimliği; bunun yerine `R2_ENDPOINT_URL` de kullanılabilir. |
| `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | Mevcut R2 yetkileri. |
| `R2_PUBLIC_BASE_URL` | Yayındaki görsel alan adı. |
| `OPENAI_BASE_URL` | Örneğin aynı Docker ağında `http://9router:20128/v1`. |
| `OPENAI_API_KEY` | 9router’da oluşturduğunuz API anahtarı. |
| `OPENAI_MODEL` | Doğrudan `cx/gpt-6-luna` veya 9router’da tanımlı combo’nun tam adı. |
| `WEB_INTERNAL_URL`, `INTERNAL_API_KEY` | Web uygulamasındaki önbellek yenileme adresi ve aynı gizli anahtar. Kullanılmıyorsa ikisini de boş bırakın. |
| `WORKER_DOCKER_NETWORK` | PostgreSQL / 9router container’larının bağlı olduğu mevcut Docker ağı. |

Container içindeki `localhost` / `127.0.0.1`, container’ın kendisidir.
Veritabanı aynı ağdaysa servis adını ve iç portunu kullanın (genellikle 5432).
9router’ın başka bir hesaba veya kotası dolu hesaba bağlı olmadığını kontrol edin.

`WORKER_DOCKER_NETWORK=trendmanga` kullanacaksanız ağ yoksa oluşturun:

```sh
docker network inspect trendmanga >/dev/null 2>&1 || docker network create trendmanga
```

9router ayrı bir container ise bu ağa da bağlanmalıdır. Örnek (adı kendi
container adınızla değiştirin; sürekli kullanım için onun Compose dosyasına da ağı ekleyin):

```sh
docker network connect --alias 9router trendmanga YOUR_9ROUTER_CONTAINER
```

9router sunucu üzerinde Docker dışında çalışıyorsa `OPENAI_BASE_URL` değerini
`http://host.docker.internal:20128/v1` yapın. Compose host-gateway eşlemesini ekler.
9router yalnızca 127.0.0.1’de dinliyorsa container erişemez; Docker’dan erişilebilen
bir arayüzde dinlemelidir. Docker üzerinden bağlantıda aynı özel ağı tercih edin.

## Çalıştırma

```sh
docker compose --env-file .env.server -f compose.cpu.yml config --quiet
docker compose --env-file .env.server -f compose.cpu.yml up -d --build
docker compose --env-file .env.server -f compose.cpu.yml logs -f --tail=100 translation-worker
```

İlk build CPU-only PyTorch ve Qt bağımlılıklarını kurar; büyük indirmeler nedeniyle
uzayabilir. Build sırasında CPU runtime, gerçek pipeline importları ve fontsuz
sunucuda Qt font yüklemesi kontrol edilir. Gerçek DB / R2 / model çağrısı yapılmaz.
İlk çeviride tespit ve temizleme modelleri indirilir; sonraki açılışlarda kalıcı
`translation-work` volume’ünden kullanılır. Bölüm iş logları da burada saklanır.
İşler R2’ye gönderilir ve mevcut admin panelindeki kuyrukta görünür.

Worker kimliği container adına göre `translator-server-...` oluşturulur.
Bilgisayardaki worker’lar ile aynı veritabanını kullanabilir: iş sahipliği kilidi
aynı kuyruğa bağlı çalışanların birbirinin işini almasını önler. Sunucu CPU’sunu
korumak için bu Compose ile başlangıçta yalnızca tek container çalıştırın.

## Kaynak sınırları ve durdurma

`TRANSLATION_CPUS=2.0` ve `TRANSLATION_MEMORY=8g` gerçek container sınırlarıdır.
CPU iç paralelliği 1 iş, 2 hesaplama thread’i ile başlar. Sınırları değiştirince
`up -d` komutunu tekrar çalıştırın. Panel üzerinden yalnız Dockerfile ile deploy
ediyorsanız CPU/RAM sınırlarını panelde ayrıca 2 CPU / 8 GB ayarlayın; Dockerfile
tek başına bu sınırları uygulayamaz. `.env.server.example` değerlerini panelin
environment alanına girin; kalıcı `/app/work` volume’ünü ve özel ağı da ekleyin.
Dockerfile yolu `Dockerfile`, build context depo köküdür; HTTP domain/port gerekmez.

```sh
docker compose --env-file .env.server -f compose.cpu.yml stop
```

Durdurma isteği yeni iş alınmasını engeller ve mevcut işin tamamlanmasını bekler;
30 dakikada tamamlanmazsa Docker işlemi kapatır. Proje işi bütün projeyi kapsayabilir.
Zorla kapanan iş sonraki worker başlangıcında mevcut retry mekanizmasıyla kurtarılır.
`down -v` kullanmayın; bu seçenek model önbelleğini ve iş loglarını da siler.

Gerçek `.env.server` ve erişim anahtarları Git’e veya Docker imajına eklenmez.
