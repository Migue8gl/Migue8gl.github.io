---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Miguel García
description: Hola, soy Miguel García. En este blog escribiré sobre *Machine Learning* y de vez en cuando de otras cosas que me interesen.
---

{% assign totalPosts = site.posts | size %}
{% assign totalWords = 0 %}

{% for post in site.posts %}
  {% assign postWords = post.content | number_of_words %}
  {% assign totalWords = totalWords | plus: postWords %}
{% endfor %}

# Hola, soy Miguel

Hi, welcome to my website! I use this tiny part of the internet to share my thoughts about machine learning, maths, cooking, philosophy, and other random things. I write for several reasons that I've explained [here](http://alexmolas.com/2023/07/15/nobody-cares-about-your-blog.html). Since 2020 I've wrote {{ totalPosts }} posts and {{ totalWords }} words. 

I try to keep the site style quite minimal because I want to focus on the content and not on the container. Some people have told me it's too minimal, but I don't care. If you like it you can find the source code [in my repo](https://github.com/alexmolas/alexmolas.github.io/). Feel free to use it as you prefer.


If you want to hire me here you have my <a href="/cv/">CV</a>.


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

---