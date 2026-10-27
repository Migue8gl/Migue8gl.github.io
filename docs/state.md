# Estado

- Blog Jekyll (tema minima con layouts propios) alojado en GitHub Pages con dominio propio `migue8gl.dev` desde el 22-09-2026 (`CNAME`).
- Sin workflow de Actions: GitHub Pages compila con su Jekyll 3.x, aunque el `Gemfile` local fija Jekyll 4.4.
- Redirecciones (verificadas el 27-10-2026): `http://`, `www.` y `migue8gl.github.io` hacen 301 a `https://migue8gl.dev/`. Es correcto y es lo que Search Console lista como «Página con redirección».
- URLs canónicas: los posts usan `/AAAA/MM/DD/slug.html`. GitHub Pages sirve también la versión sin `.html` (200), que tiene la etiqueta canónica hacia la versión con `.html`.
- `jekyll-sitemap` genera `sitemap.xml` y `robots.txt` con URLs del dominio nuevo.
- `docs/` está excluida del build en `_config.yml`.
