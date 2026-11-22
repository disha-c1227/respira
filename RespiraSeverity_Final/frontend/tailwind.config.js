/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#f0f5fa',
          100: '#e1ecf5',
          200: '#c2d9eb',
          500: '#1d5a8a',
          700: '#133a59',
          800: '#0f2942',
          900: '#0a1d2f',
        },
        teal: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#008080',
          900: '#134e4a',
        }
      }
    },
  },
  plugins: [],
}
