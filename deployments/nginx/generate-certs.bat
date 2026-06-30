@echo off
REM Genere un certificat SSL auto-signe pour localhost (valide 365 jours)
REM Necessite Docker

docker run --rm -v "%CD%\certs:/certs" alpine/openssl req -x509 -nodes -days 365 -newkey rsa:2048 ^
  -keyout /certs/nginx.key ^
  -out /certs/nginx.crt ^
  -subj "/CN=localhost"

echo Certificats generes dans deployments\nginx\certs\
