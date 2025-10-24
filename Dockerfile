FROM node:23-alpine

COPY entrypoint.sh /

WORKDIR /slidev

RUN chmod +x /entrypoint.sh

ENV CHOKIDAR_USEPOLLING=true

ENTRYPOINT [ "/entrypoint.sh" ]