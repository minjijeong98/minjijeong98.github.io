document.addEventListener("nav", () => {
  const floatingToc = document.querySelector<HTMLElement>(".floating-toc")
  if (!floatingToc) return

  const segments = floatingToc.querySelectorAll<HTMLElement>(".toc-segment")
  const links = floatingToc.querySelectorAll<HTMLAnchorElement>(".toc-panel a")
  const headings = Array.from(links)
    .map((a) => {
      const id = a.getAttribute("href")?.slice(1)
      return id ? document.getElementById(id) : null
    })
    .filter((el): el is HTMLElement => el !== null)

  // Activate segment on scroll
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const id = entry.target.id
          const idx = headings.findIndex((h) => h.id === id)
          if (idx === -1) return
          segments.forEach((s) => s.classList.remove("active"))
          segments[idx]?.classList.add("active")
          links.forEach((l) => l.classList.remove("active"))
          links[idx]?.classList.add("active")
        }
      }
    },
    { rootMargin: "-10% 0px -80% 0px", threshold: 0 },
  )

  headings.forEach((h) => observer.observe(h))
  window.addCleanup(() => headings.forEach((h) => observer.unobserve(h)))

  // Smooth scroll on link click
  links.forEach((link) => {
    const handler = (e: Event) => {
      e.preventDefault()
      const id = (link as HTMLAnchorElement).getAttribute("href")?.slice(1)
      if (!id) return
      const target = document.getElementById(id)
      target?.scrollIntoView({ behavior: "smooth", block: "start" })
    }
    link.addEventListener("click", handler)
    window.addCleanup(() => link.removeEventListener("click", handler))
  })
})
