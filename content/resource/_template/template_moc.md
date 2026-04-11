---
title: <% tp.file.title %>
created: <% tp.file.creation_date() %>
updated: <% tp.date.now("YYYY-MM-DD HH:mm") %>
tags:
  - moc
status: 🌱seed
draft: true
---

> TL;DR
> 

## 🗺️ Map of Contents
- 

## 📥 미분류/수집된 노트
```dataview
list from #note
where contains(file.name, this.file.name) or contains(tags, "특정주제")
limit 10