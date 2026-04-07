import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        blsNavy: '#0f2d52',
        blsTeal: '#0f766e',
        blsSand: '#f6f3eb'
      },
      boxShadow: {
        widget: '0 10px 30px rgba(15, 45, 82, 0.15)'
      }
    }
  },
  plugins: []
};

export default config;
