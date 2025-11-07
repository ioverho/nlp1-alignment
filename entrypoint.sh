#!/bin/sh

echo "Entered startup script"

if [ "$NPM_MIRROR" != "" ]; then
    npm config set registry $NPM_MIRROR
fi

echo "Running NPM install..."

npm install @slidev/cli@v52.1.0 @slidev/theme-default
npm install -D playwright-chromium
npm install slidev-theme-neversink

if [ -f /slidev/slides.md ]; then
    echo "Start slidev..."

else
    echo "slides.md not found in the bind mount to /slidev"
    cp -f /slidev/node_modules/@slidev/cli/template.md /slidev/slides.md
    sed -i ':a;N;$!ba;s/GitHub"\n/GitHub"/g' /slidev/slides.md

fi

if [ "$NPM_MIRROR" != "" ]; then
    npm config delete registry
fi

npx slidev --remote