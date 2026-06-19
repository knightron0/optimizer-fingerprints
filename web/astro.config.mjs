// @ts-check
import { defineConfig } from 'astro/config';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1];
const owner = process.env.GITHUB_REPOSITORY_OWNER;
const isPagesBuild = process.env.ASTRO_GITHUB_PAGES === 'true';
const isProjectPage = repository && !repository.endsWith('.github.io');
const pagesSite = repository?.endsWith('.github.io')
	? `https://${repository}`
	: owner
		? `https://${owner}.github.io`
		: undefined;

// https://astro.build/config
export default defineConfig({
	site: isPagesBuild ? pagesSite : undefined,
	base: isPagesBuild && isProjectPage ? `/${repository}` : undefined,
});
