# Инструкция по развертыванию на сервере

## Вариант 1: Использование отдельного порта (проще всего)

Если на вашем сервере уже работает сайт на порту 80, запустите Docker контейнер на другом порту.

### Шаги:

1. **Клонируйте репозиторий на сервер:**
   ```bash
   git clone https://github.com/wowsandek/scud.git
   cd scud
   git checkout скуд
   ```

2. **Создайте файл `.env` с другим портом:**
   ```bash
   cp env.example .env
   ```
   
   Отредактируйте `.env` и укажите свободный порт (например, 8080):
   ```
   NGINX_PORT=8080
   ```

3. **Запустите контейнер:**
   ```bash
   docker-compose up -d --build
   ```

4. **Проверьте работу:**
   ```bash
   docker-compose ps
   docker-compose logs -f firesec-ui
   ```

5. **Доступ к приложению:**
   - По адресу: `http://ваш-сервер:8080`
   - Или через домен с портом: `http://firesec.ваш-домен:8080`

---

## Вариант 2: Reverse Proxy через существующий Nginx (рекомендуется)

Если у вас уже настроен Nginx на сервере, можно настроить reverse proxy для работы на стандартном порту 80 через поддомен или путь.

### Шаги:

1. **Запустите Docker контейнер на внутреннем порту:**
   
   Создайте `.env`:
   ```
   NGINX_PORT=8080
   ```
   
   Запустите:
   ```bash
   docker-compose up -d --build
   ```

2. **Добавьте конфигурацию в ваш основной Nginx:**

   **Вариант A: Через поддомен** (например, `firesec.ваш-домен.com`)
   
   Создайте файл `/etc/nginx/sites-available/firesec`:
   ```nginx
   server {
       listen 80;
       server_name firesec.ваш-домен.com;
       
       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```
   
   Активируйте:
   ```bash
   sudo ln -s /etc/nginx/sites-available/firesec /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

   **Вариант B: Через путь** (например, `ваш-домен.com/firesec`)
   
   Добавьте в существующий конфиг Nginx:
   ```nginx
   location /firesec {
       rewrite ^/firesec/?(.*) /$1 break;
       proxy_pass http://127.0.0.1:8080;
       proxy_http_version 1.1;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       proxy_set_header X-Forwarded-Proto $scheme;
   }
   ```
   
   ⚠️ **Важно:** Для работы через путь может потребоваться настроить `base` в `vite.config.ts` и пересобрать контейнер.

3. **Настройте DNS** (если используете поддомен):
   ```
   A запись: firesec -> IP вашего сервера
   ```

---

## Вариант 3: Использование существующего Docker сети

Если у вас уже есть другие Docker контейнеры, можно подключить к общей сети.

### Шаги:

1. **Найдите имя существующей сети:**
   ```bash
   docker network ls
   ```

2. **Обновите `docker-compose.yml`:**
   
   Замените секцию `networks`:
   ```yaml
   networks:
     default:
       external:
         name: имя_существующей_сети
   ```

3. **Запустите:**
   ```bash
   docker-compose up -d --build
   ```

---

## Проверка работы

После развертывания проверьте:

1. **Статус контейнера:**
   ```bash
   docker-compose ps
   ```

2. **Логи:**
   ```bash
   docker-compose logs -f firesec-ui
   ```

3. **Health check:**
   ```bash
   curl http://localhost:8080/health
   ```

4. **Доступность из браузера:**
   - Откройте указанный адрес в браузере
   - Должен загрузиться интерфейс FireSec API UI

---

## Обновление приложения

Когда появятся новые изменения:

```bash
cd scud
git pull origin скуд
docker-compose down
docker-compose up -d --build
```

---

## Устранение проблем

### Порт уже занят
Если порт занят, измените `NGINX_PORT` в `.env` на другой свободный порт.

### Контейнер не запускается
Проверьте логи:
```bash
docker-compose logs firesec-ui
```

### Nginx не проксирует запросы
Убедитесь, что:
- Docker контейнер запущен и слушает на указанном порту
- В конфиге Nginx указан правильный порт (8080)
- Выполнили `sudo nginx -t` и `sudo systemctl reload nginx`

---

## Безопасность

Для production рекомендуется:

1. **Настроить SSL/HTTPS** через Let's Encrypt:
   ```bash
   sudo certbot --nginx -d firesec.ваш-домен.com
   ```

2. **Ограничить доступ** через firewall (если нужно):
   ```bash
   sudo ufw allow 8080/tcp
   ```

3. **Использовать переменные окружения** для чувствительных данных (если добавите их в будущем)

