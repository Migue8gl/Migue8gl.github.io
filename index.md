---
layout: default
title: Miguel García
description: Hola, soy Miguel García. En este blog escribiré sobre *Machine Learning* y, de vez en cuando, sobre otras cosas que me interesen.
---

<div class="home-intro">
  <h1>Hola, soy Miguel</h1>

  <img
    src="/assets/images/home/hero.png"
    alt="Miguel García – Ciencia de datos y Machine Learning"
    class="home-separator"
  />
  
  <p>¡Bienvenidos a mi blog personal! La verdad es que llevo ya un tiempo pensando en escribir algo, ya sea en una libretilla o en un blog. Siempre me han interesado los blogs y los foros, pues considero que es la parte más real que nos queda en este submundo virtual, por desgracia.</p>
  
  <p>Los contenidos SEO-optimizados, la multimedia generada por inteligencia artificial y completamente automatizado están dejando un espacio virtual bastante frío. Es en los blogs personales, foros con comunidades interesantes y con personas que participan activamente, donde se devuelve la magia que Internet siempre ha tenido y poco a poco pierde.</p>
  
  <p>Así que iré escribiendo poco a poco por aquí sobre ciencia de datos, algoritmos, aprendizaje automático y conceptos de ese mundillo.</p>
  
  <p>Además, de vez en cuando voy subiendo a mi <a href="https://github.com/Migue8gl">Github</a> proyectos personales de los que hablaré por aquí.</p>
  
  <p>Dejo por aquí mi <a href="https://migue8gl.github.io/cv">CV</a> con un resumen de mi experiencia laboral.</p>
</div>

<div class="home-posts">
  <h2>Últimos posts</h2>
  
  <ul class="post-list">
    {% assign postCount = 0 %}
    {% for post in site.posts %}
    {% if postCount < 5 %}
    <li class="post-item">
      <time class="post-date">{{ post.date | date: "%b %d %Y" }}</time>
      <span class="separator">·</span>
      <a href="{{ post.url }}" class="post-link">{{ post.title }}</a>
    </li>
    {% assign postCount = postCount | plus: 1 %}
    {% endif %}
    {% endfor %}
  </ul>
</div>