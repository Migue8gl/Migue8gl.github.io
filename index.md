---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Miguel García
description: Hola, soy Miguel García. En este blog escribiré sobre *Machine Learning* y de vez en cuando de otras cosas que me interesen.
---

# Hola, soy Miguel

¡Bienvenidos a mi blog personal! La verdad es que llevo ya un tiempo pensando en escribir algo, ya sea en una libretilla o en un blog, y es cierto que siempre me han interesado los blogs y los foros, pues considero que es la parte más real que nos queda en este submundo virtual, por desgracia. 

Los contenidos SEO-optimizados, multimedia generada por inteligencia artificial y completamente automatizado están dejando un espacio virtual bastante frío. Es en los blogs personales, foros con comunidades interesantes y con personas que participan activamente, donde se devuelve la magia que Internet siempre ha tenido y poco a poco pierde.

Asi que iré escribiendo poco a poco por aquí sobre ciencia de datos, algoritmos, aprendizaje automático y en general conceptos de ese mundillo.

Además, de vez en cuando voy subiendo a mi <a href="https://github.com/Migue8gl">Github</a> proyectos personales de los que hablaré por aquí.

Dejo por aquí mi <a href="/cv/">CV</a> con un resumen de mi experiencia laboral.


---

# Últimos posts

{% assign postCount = 0 %}
{% for post in site.posts %}
{% if postCount < 5 %}
  <li>
    <span class="post-date">{{ post.date | date: "%b %d %Y" }}</span> · <a href="{{ post.url }}">{{ post.title }}</a>
  </li>
  {% assign postCount = postCount | plus: 1 %}
{% endif %}
{% endfor %}
