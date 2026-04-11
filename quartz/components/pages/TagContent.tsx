import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/listPage.scss"
import tagBrowserStyle from "../styles/tagBrowser.scss"
import { PageList, SortFn } from "../PageList"
import { FullSlug, getAllSegmentPrefixes, resolveRelative, simplifySlug } from "../../util/path"
import { QuartzPluginData } from "../../plugins/vfile"
import { Root } from "hast"
import { htmlToJsx } from "../../util/jsx"
import { i18n } from "../../i18n"
import { ComponentChildren } from "preact"
import { concatenateResources } from "../../util/resources"
// @ts-ignore
import tagBrowserScript from "../scripts/tagBrowser.inline"

interface TagContentOptions {
  sort?: SortFn
  numPages: number
}

const defaultOptions: TagContentOptions = {
  numPages: 10,
}

export default ((opts?: Partial<TagContentOptions>) => {
  const options: TagContentOptions = { ...defaultOptions, ...opts }

  const TagContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { tree, fileData, allFiles, cfg } = props
    const slug = fileData.slug

    if (!(slug?.startsWith("tags/") || slug === "tags")) {
      throw new Error(`Component "TagContent" tried to render a non-tag page: ${slug}`)
    }

    const tag = simplifySlug(slug.slice("tags/".length) as FullSlug)
    const allPagesWithTag = (tag: string) =>
      allFiles.filter((file) =>
        (file.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes).includes(tag),
      )

    const content = (
      (tree as Root).children.length === 0
        ? fileData.description
        : htmlToJsx(fileData.filePath!, tree)
    ) as ComponentChildren
    const cssClasses: string[] = fileData.frontmatter?.cssclasses ?? []
    const classes = cssClasses.join(" ")

    if (tag === "/") {
      // Build tag metadata sorted by most recently modified article (default)
      const allTags = [
        ...new Set(
          allFiles.flatMap((data) => data.frontmatter?.tags ?? []).flatMap(getAllSegmentPrefixes),
        ),
      ]
      const tagItemMap: Map<string, QuartzPluginData[]> = new Map()
      for (const t of allTags) {
        tagItemMap.set(t, allPagesWithTag(t))
      }

      const tagMeta = allTags.map((t) => {
        const pages = tagItemMap.get(t)!
        const latestDate = pages.reduce((max, page) => {
          const d = page.dates?.modified ?? page.dates?.created
          return d && d > max ? d : max
        }, new Date(0))
        return { tag: t, count: pages.length, latestDate: latestDate.getTime() }
      })

      // Sort by most recent by default
      tagMeta.sort((a, b) => b.latestDate - a.latestDate)

      return (
        <div class="popover-hint">
          <div class="tag-browser">
            <div class="tag-browser-header">
              <span class="tag-total">{i18n(cfg.locale).pages.tagContent.totalTags({ count: allTags.length })}</span>
              <div class="tag-sort-controls">
                <button class="tag-sort-btn active" data-sort="recent">최근 추가순</button>
                <button class="tag-sort-btn" data-sort="count">언급 많은 순</button>
              </div>
            </div>
            <ul id="tag-browser-list">
              {tagMeta.map(({ tag: t, count, latestDate }) => {
                const pages = tagItemMap.get(t)!
                const tagListingPage = `/tags/${t}` as FullSlug
                const href = resolveRelative(fileData.slug!, tagListingPage)
                return (
                  <li class="tag-item" data-count={count} data-latest={latestDate}>
                    <button class="tag-header">
                      <span class="tag-name">{t}</span>
                      <span class="tag-count">{count}</span>
                      <span class="tag-chevron">▸</span>
                    </button>
                    <ul class="tag-articles">
                      {pages.map((page) => (
                        <li>
                          <a href={resolveRelative(fileData.slug!, page.slug!)}>
                            {page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title}
                          </a>
                          {page.dates?.modified && (
                            <span class="tag-article-date">
                              {page.dates.modified.toLocaleDateString("ko-KR", {
                                year: "numeric", month: "short", day: "numeric",
                              })}
                            </span>
                          )}
                        </li>
                      ))}
                      <li class="tag-see-all">
                        <a href={href}>전체 보기 →</a>
                      </li>
                    </ul>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      )
    } else {
      const pages = allPagesWithTag(tag)
      const listProps = { ...props, allFiles: pages }

      return (
        <div class="popover-hint">
          <article class={classes}>{content}</article>
          <div class="page-listing">
            <p>{i18n(cfg.locale).pages.tagContent.itemsUnderTag({ count: pages.length })}</p>
            <div>
              <PageList {...listProps} sort={options?.sort} />
            </div>
          </div>
        </div>
      )
    }
  }

  TagContent.afterDOMLoaded = tagBrowserScript
  TagContent.css = concatenateResources(style, PageList.css, tagBrowserStyle)
  return TagContent
}) satisfies QuartzComponentConstructor
