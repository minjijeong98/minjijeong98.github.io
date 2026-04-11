import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// @ts-ignore
import script from "./scripts/floatingToc.inline"
import style from "./styles/floatingToc.scss"

const FloatingTOC: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const toc = fileData.toc
  if (!toc || toc.length === 0) return null

  return (
    <div class="floating-toc">
      <div class="toc-bar">
        {toc.map((_entry, i) => (
          <div class="toc-segment" data-index={i} />
        ))}
      </div>
      <nav class="toc-panel">
        <p class="toc-panel-title">목차</p>
        <ul>
          {toc.map((entry) => (
            <li style={`padding-left: ${entry.depth * 10}px`}>
              <a href={`#${entry.slug}`}>{entry.text}</a>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  )
}

FloatingTOC.afterDOMLoaded = script
FloatingTOC.css = style

export default (() => FloatingTOC) satisfies QuartzComponentConstructor
