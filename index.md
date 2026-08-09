---
layout: default
title: Miguel García
description: Hola, soy Miguel García. En este blog escribiré sobre Machine Learning, Data Science y, de vez en cuando, sobre otras cosas que me produzcan curiosidad. Disfruta de la lectura :)
---

<div class="home-intro">
  <h1>Hola, soy Miguel</h1>

  <img
    src="/assets/images/home/hero_2.png"
    alt="Miguel García – Ciencia de datos y Machine Learning"
    class="home-separator"
  />
  
  <p>¡Bienvenidos a mi blog personal! Este proyecto nació de mi necesidad de expresión. Al escribir nos forzamos a organizar las ideas, elegir una narrativa con la que expresarnos y además, nos obliga a reflexionar y corregir sobre lo ya escrito. Es algo que se hace cada vez menos y no quiero perder esa cadena de pensamiento que tanto aporta.</p>
  
  <p>Así que iré escribiendo poco a poco por aquí sobre ciencia de datos, algoritmos, aprendizaje automático y conceptos de ese mundillo que tanto me apasiona y al que tengo la suerte de dedicarme profesionalmente.</p>
  
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