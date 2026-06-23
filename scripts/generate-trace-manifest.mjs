import { readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdir } from 'node:fs/promises';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const traceDirectory = resolve(repoRoot, process.argv[2] ?? 'traces');
const outputPath = resolve(repoRoot, process.argv[3] ?? 'web/public/traces-manifest.json');
const traceUrlPrefix = process.env.TRACE_URL_PREFIX ?? process.argv[4] ?? './traces/';
const displayNames = JSON.parse(await readFile(resolve(repoRoot, 'scripts/trace-display-names.json'), 'utf8'));

function fallbackDisplayName(runName) {
	const boilerplate = new Set(['entry', 'submission', 'with', 'aux', 'adam', 'lr', 'wd', 'fol']);
	const tokens = runName
		.split('_')
		.filter((token) => token && !/^20\d{6}$/.test(token) && !/^\d+$/.test(token) && !/^[a-f0-9]{8,}$/.test(token))
		.filter((token) => !boilerplate.has(token.toLowerCase()));
	return tokens.slice(0, 2).map((token) => token.replace(/^./, (letter) => letter.toUpperCase())).join(' ') || 'Trace';
}

const filenames = (await readdir(traceDirectory))
	.filter((filename) => filename.endsWith('.json'))
	.sort();

const traces = [];
for (const filename of filenames) {
	const trace = JSON.parse(await readFile(resolve(traceDirectory, filename), 'utf8'));
	if (trace.schema !== 'nanogpt_optimizer_trace') {
		console.warn(`Skipping ${filename}: unsupported schema`);
		continue;
	}

	const optimizerClasses = [...new Set(trace.optimizer_classes ?? [])];
	const runName = trace.run_name || filename.replace(/\.json$/, '');
	const title = displayNames[runName] ?? fallbackDisplayName(runName);
	const details = [];
	if (title !== runName) details.push(runName);
	if (optimizerClasses.length) details.push(optimizerClasses.join(' + '));
	if (Number.isFinite(trace.completed_steps)) {
		details.push(`${trace.completed_steps.toLocaleString('en-US')} completed steps`);
	}
	if (Number.isFinite(trace.snapshot_interval)) {
		details.push(`snapshots every ${trace.snapshot_interval.toLocaleString('en-US')} steps`);
	}
	const id = filename.replace(/\.json$/, '');

	traces.push({
		id,
		title,
		description: details.join(' · '),
		trace_url: `${traceUrlPrefix.replace(/\/?$/, '/')}${encodeURIComponent(filename)}`,
	});
}

const duplicateTitles = traces
	.map(({ title }) => title)
	.filter((title, index, titles) => titles.indexOf(title) !== index);
if (duplicateTitles.length) {
	throw new Error(`Manifest titles must be unique: ${[...new Set(duplicateTitles)].join(', ')}`);
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify({ traces }, null, 2)}\n`);
console.log(`Wrote ${traces.length} traces to ${outputPath}`);
