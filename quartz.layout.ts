import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/minjijeong98",
      LinkedIn: "https://www.linkedin.com/in/minji-jeong-2396a4300",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.ArticleTitle(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.ContentMeta(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.TagList(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.RecentNotes({
        title: "최근 게시물",
        limit: 100,
        showTags: true,
      }),
      condition: (page) => page.fileData.slug === "index",
    }),
    Component.ConditionalRender({
      component: Component.FloatingTOC(),
      condition: (page) =>
        page.fileData.slug !== "index" && (page.fileData.toc?.length ?? 0) > 0,
    }),
  ],
  left: [
    Component.SidebarHeader(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search({ enablePreview: false }),
    Component.Explorer(),
    Component.Graph(),
  ],
  right: [],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.SidebarHeader(),
    Component.MobileOnly(Component.Spacer()),
    Component.Search({ enablePreview: false }),
    Component.Explorer(),
    Component.Graph(),
  ],
  right: [],
}
