# Backlog

## Alta prioridad

- [x] Quitar `<script src="https://polyfill.io/...">` y su `preconnect` de `_layouts/default.html`. El dominio polyfill.io sufrió un ataque a la cadena de suministro en 2024; hoy devuelve 404, pero quien lo controla podría volver a servir código. MathJax 3 no lo necesita en navegadores actuales.

## Search Console / SEO (migración a migue8gl.dev, 22-09-2026)

- [x] Los enlaces internos a posts, `/about` y `/cv` no llevaban `.html`; Google los descubría y los marcaba como «Página alternativa con etiqueta canónica adecuada». Ahora apuntan a la URL canónica.
- [ ] En Search Console, inspeccionar y «Solicitar indexación» de los posts que aparezcan en «Descubierta/Rastreada: actualmente sin indexar».
- [ ] Si la propiedad antigua `https://migue8gl.github.io` está verificada en Search Console, usar la herramienta «Cambio de dirección» hacia `migue8gl.dev`.
- [ ] Revisar a mediados de noviembre de 2026 si el número de páginas indexadas ha subido.

## Ideas

- [ ] Plantearse URLs sin `.html` (`permalink: /:year/:month/:day/:title`) cuando el dominio esté asentado. Cambiaría la URL canónica de todos los posts, así que mejor no hacerlo en plena migración.
- [ ] Añadir `author` a `_config.yml`: el JSON-LD de los posts usa `site.author`, que ahora está vacío.
- [ ] Comprobar el mismo problema al escribir posts nuevos: enlazar siempre a `/AAAA/MM/DD/slug.html`.
