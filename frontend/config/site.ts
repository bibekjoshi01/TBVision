export type SiteConfig = typeof siteConfig;

export const siteConfig = {
  name: "TBVision AI",
  description: "Advanced medical AI system for chest X-ray analysis.",
  navItems: [
    {
      label: "Home",
      href: "/",
    },
    {
      label: "Analysis",
      href: "/analysis",
    },
  ],
  navMenuItems: [
    {
      label: "Home",
      href: "/",
    },
    {
      label: "Analysis",
      href: "/analysis",
    },
  ],
  links: {
    github: "https://github.com",
    docs: "/",
  },
};
