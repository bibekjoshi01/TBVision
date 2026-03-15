'use client'

import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

import React, { useState } from 'react'
import { Copy, Check } from 'lucide-react'

type Props = {
  content?: string
}

const MarkdownRenderer: React.FC<Props> = ({ content = '' }) => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  // const cleaned = content?.replace(/^\s+/gm, '')

  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text)
    setCopiedCode(text)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  return (
    <div className="w-full max-w-4xl mx-auto px-6 text-base leading-7 text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: true }]]}
        rehypePlugins={[rehypeKatex]}
        components={{
          /* Headings */
          h1: ({ ...props }) => (
            <h1
              className="mt-10 mb-6 text-3xl font-semibold tracking-tight leading-tight"
              {...props}
            />
          ),

          h2: ({ ...props }) => (
            <h2
              className="mt-8 mb-4 text-2xl font-semibold tracking-tight leading-snug"
              {...props}
            />
          ),

          h3: ({ ...props }) => (
            <h3 className="mt-6 mb-2 text-xl font-medium leading-snug" {...props} />
          ),

          /* Paragraph  */
          p: ({ ...props }) => (
            <p className="mb-6 text-foreground/90 text-base leading-8" {...props} />
          ),

          /* Links */
          a: ({ ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4 hover:opacity-80 transition"
            />
          ),

          table: ({ ...props }) => (
            <div className="w-full overflow-x-auto my-8 border border-border rounded-lg">
              <table className="w-full border-collapse text-sm" {...props} />
            </div>
          ),

          thead: ({ ...props }) => <thead className="bg-muted" {...props} />,

          tbody: ({ ...props }) => <tbody {...props} />,

          tr: ({ ...props }) => (
            <tr
              className="border-b border-border last:border-b-0 hover:bg-muted/40 transition"
              {...props}
            />
          ),

          th: ({ ...props }) => (
            <th
              className="px-5 py-3 text-left font-semibold border-r border-border last:border-r-0 whitespace-nowrap"
              {...props}
            />
          ),

          td: ({ ...props }) => (
            <td
              className="px-5 py-3 border-r border-border last:border-r-0 text-foreground/80"
              {...props}
            />
          ),

          /* Code */
          code({ inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            const code = String(children).replace(/\n$/, '')

            if (!inline && match) {
              const isCopied = copiedCode === code

              return (
                <div className="group relative my-8 rounded-xl overflow-hidden text-sm border border-zinc-700">
                  {/* Header */}
                  <div className="flex items-center justify-between px-4 py-2 bg-[#282c34] border-b border-zinc-700">
                    <span className="text-xs uppercase tracking-wide text-zinc-400">
                      {match[1]}
                    </span>

                    <button
                      onClick={() => handleCopy(code)}
                      className="h-7 w-7 cursor-pointer text-zinc-400 hover:text-white transition"
                    >
                      {isCopied ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </button>
                  </div>

                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{ margin: 0, borderTopLeftRadius: 0, borderTopRightRadius: 0 }}
                  >
                    {code}
                  </SyntaxHighlighter>
                </div>
              )
            }

            return (
              <code
                className="px-1.5 py-0.5 rounded-md bg-muted/60 text-sm font-mono border border-border"
                {...props}
              >
                {children}
              </code>
            )
          },

          /* Blockquote */
          blockquote: ({ ...props }) => (
            <blockquote
              className="my-6 border-l-4 border-border bg-muted/50 pl-4 py-2 rounded-r-md text-foreground/80"
              {...props}
            />
          ),

          /* Lists */
          ul: ({ ...props }) => (
            <ul className="list-disc ml-6 my-6 space-y-2 text-foreground/85" {...props} />
          ),
          ol: ({ ...props }) => <ol className="list-decimal ml-6 my-4 space-y-1.5" {...props} />,
          li: ({ ...props }) => <li className="leading-relaxed" {...props} />,

          /* Images */
          img: ({ ...props }) => (
            <img {...props} className="rounded-lg border border-border my-6 max-w-full" />
          ),

          /* Horizontal Rule */
          hr: () => <hr className="my-8 border-border/60" />,
        }}
      >
        {content || 'No content available.'}
      </ReactMarkdown>
    </div>
  )
}

export default MarkdownRenderer
