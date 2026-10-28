# Parul Aggarwal — Portfolio Website

A personal portfolio website for **Parul Aggarwal**, Software Development Engineer (SDE). Inspired by modern minimalist developer portfolios with dark/light themes, rotating badges, and interactive components.

---

## ✨ Features

- **Iconic Visual Aesthetic**: Clean, typography-focused layout inspired by top developer portfolios.
- **Bespoke Artwork**: Watercolor sketch portrait with subtle floating hover animations.
- **Dark & Light Mode**: Instant smooth theme switching with persistent state saved in `localStorage`.
- **Interactive "Hire Me" Rotating Badge**: Infinite smooth SVG curved-text rotation around a quick-action button.
- **Interactive Lightbulb Idea Widget**: Floating sticker widget delivering engineering spark tips.
- **Interactive Resume Modal**: Built-in resume viewer with print/save as PDF support.
- **Featured Projects Showcase**: High-impact cards with live demo triggers, GitHub links, and tech stack tags.
- **One-Click Email Copy**: Direct email copy to clipboard with an animated toast alert.
- **Contact Form**: Responsive message form with client-side feedback.
- **Mobile Responsive**: Flex & CSS Grid layout optimized for smartphones, tablets, and desktop displays.

---

## 🚀 Quick Start

To preview locally:

### Option 1: Direct in Browser
Double-click `index.html` in your file explorer or open it in Chrome, Safari, or Firefox:
```bash
open index.html
```

### Option 2: Local HTTP Server (Recommended)
Run Python's built-in server:
```bash
python3 -m http.server 4321
```
Then visit [http://localhost:4321](http://localhost:4321) in your browser.

---

## 📁 File Structure

```
PortFolio/
├── index.html                 # Main website HTML with semantic layout and SEO
├── styles/
│   └── main.css               # Design system, CSS variables, dark/light themes & responsiveness
├── scripts/
│   └── main.js                # Theme switcher, modals, toast notifications, animations
├── assets/
│   └── images/
│       ├── hero-avatar.jpg            # Watercolor software engineer portrait
│       ├── lightbulb.jpg              # Stylized idea lightbulb icon
│       ├── project-observability.jpg  # Project preview: Cloud Observability Platform
│       └── project-devplatform.jpg    # Project preview: DevHub Cloud Workspace
└── README.md
```

---

## 🎨 Personalization Guide

1. **Social Links**: In [index.html](file:///Users/parulaggarwal/Desktop/PortFolio/index.html#L42-L65), update the `href` attributes for your GitHub, LinkedIn, Twitter/X, and LeetCode handles.
2. **Projects**: Edit or add project cards in the `#projects` section of [index.html](file:///Users/parulaggarwal/Desktop/PortFolio/index.html#L140-L245).
3. **Resume Details**: Update the modal in [index.html](file:///Users/parulaggarwal/Desktop/PortFolio/index.html#L330-L400) with your actual company experiences, degrees, and milestones.
