document.addEventListener("nav", () => {
  const list = document.getElementById("tag-browser-list")
  if (!list) return

  // Accordion toggle
  const headers = list.querySelectorAll<HTMLButtonElement>(".tag-header")
  headers.forEach((header) => {
    const handler = () => {
      const item = header.closest<HTMLElement>(".tag-item")
      const articles = item?.querySelector<HTMLElement>(".tag-articles")
      const chevron = header.querySelector<HTMLElement>(".tag-chevron")
      if (!articles || !chevron) return
      const isOpen = articles.classList.toggle("expanded")
      chevron.textContent = isOpen ? "▾" : "▸"
    }
    header.addEventListener("click", handler)
    window.addCleanup(() => header.removeEventListener("click", handler))
  })

  // Sort buttons
  const sortBtns = document.querySelectorAll<HTMLButtonElement>(".tag-sort-btn")
  sortBtns.forEach((btn) => {
    const handler = () => {
      sortBtns.forEach((b) => b.classList.remove("active"))
      btn.classList.add("active")
      const sortBy = btn.dataset.sort as "recent" | "count"
      const items = Array.from(list.querySelectorAll<HTMLElement>(".tag-item"))
      items.sort((a, b) => {
        if (sortBy === "count") {
          return Number(b.dataset.count) - Number(a.dataset.count)
        }
        return Number(b.dataset.latest) - Number(a.dataset.latest)
      })
      items.forEach((item) => list.appendChild(item))
    }
    btn.addEventListener("click", handler)
    window.addCleanup(() => btn.removeEventListener("click", handler))
  })
})
