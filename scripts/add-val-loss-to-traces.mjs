import fs from 'node:fs';
import path from 'node:path';

const repoRoot = path.resolve(import.meta.dirname, '..');
const defaultTsvPath = path.join(repoRoot, 'trace_log_correspondence.tsv');
const defaultTraceDir = '/Users/sarthakmangla/code/fingerprint-traces';

const args = new Set(process.argv.slice(2));
const dryRun = args.has('--dry-run');
const tsvPath = valueAfter('--tsv') ?? defaultTsvPath;
const traceDir = valueAfter('--trace-dir') ?? defaultTraceDir;

function valueAfter(flag) {
	const index = process.argv.indexOf(flag);
	return index === -1 ? null : process.argv[index + 1];
}

function parseTsv(text) {
	const [headerLine, ...lines] = text.trimEnd().split(/\r?\n/);
	const headers = headerLine.split('\t');
	return lines.map((line, lineIndex) => {
		const columns = line.split('\t');
		if (columns.length !== headers.length) {
			throw new Error(`Line ${lineIndex + 2} has ${columns.length} columns, expected ${headers.length}`);
		}
		return Object.fromEntries(headers.map((header, index) => [header, columns[index]]));
	});
}

function parseValLossSeries(value) {
	if (!value) return [];
	return value.split(';').map((entry) => {
		const [stepText, lossText] = entry.split(':');
		const step = Number(stepText);
		const loss = Number(lossText);
		if (!Number.isFinite(step) || !Number.isFinite(loss)) {
			throw new Error(`Invalid val loss point: ${entry}`);
		}
		return { step, loss };
	});
}

const rows = parseTsv(fs.readFileSync(tsvPath, 'utf8'));
const traceFiles = new Set(
	fs.readdirSync(traceDir)
		.filter((name) => name.endsWith('.json')),
);

const matched = [];
const missing = [];
const updated = [];

for (const row of rows) {
	const filename = path.posix.basename(row.trace_json);
	const series = parseValLossSeries(row.val_loss_series);
	if (!traceFiles.has(filename)) {
		missing.push({ filename, runName: row.run_name, points: series.length });
		continue;
	}

	matched.push({ filename, runName: row.run_name, points: series.length });
	const tracePath = path.join(traceDir, filename);
	const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
	trace.val_loss_series = series;
	trace.val_loss_summary = {
		final_step: Number(row.last_val_step),
		final_loss: Number(row.last_val_loss),
		points: series.length,
		source_log_file: row.log_file,
	};
	updated.push(filename);

	if (!dryRun) {
		fs.writeFileSync(tracePath, `${JSON.stringify(trace, null, 2)}\n`);
	}
}

console.log(`${dryRun ? 'Would update' : 'Updated'} ${updated.length} trace JSONs`);
console.log(`Matched TSV rows: ${matched.length}`);
console.log(`Missing target JSONs: ${missing.length}`);

if (missing.length) {
	console.log('\nMissing target JSONs:');
	for (const item of missing) {
		console.log(`- ${item.filename}\t${item.runName}\t${item.points} points`);
	}
}
