import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type MetricValue = number | null;

export interface TraceParameter {
	name: string;
	shape: number[];
	ndim: number;
	numel: number;
	optimizer_index?: number;
	group_index?: number;
	metrics: Record<string, MetricValue>;
}

export interface TraceSnapshot {
	step: number;
	learning_rates?: number[][];
	parameters: TraceParameter[];
}

export interface NanoGptTrace {
	schema: 'nanogpt_optimizer_trace';
	run_name: string;
	completed_steps: number;
	snapshot_interval: number;
	history_beta?: number;
	epsilon?: number;
	history_semantics?: string;
	world_size?: number;
	optimizer_classes: string[];
	metric_names: string[];
	snapshots: TraceSnapshot[];
}

export interface NormalizedTrace extends NanoGptTrace {
	run_id: string;
	fingerprint_id: string;
	display_name: string;
	optimizer: {
		name: string;
		family: string;
		classes: string[];
	};
	parameter_metric_names: string[];
}

export interface LoadedRun {
	run_id: string;
	path: string;
	schema: 'nanogpt_optimizer_trace';
	run_name: string;
	snapshot_count: number;
	completed_steps: number;
	optimizer_classes: string[];
	full: NormalizedTrace;
}

const webRoot = process.cwd();
const repoRoot = resolve(webRoot, '..');
const traceRoot = resolve(repoRoot, 'traces');

export function formatJsonValue(value: JsonValue | undefined): string {
	if (value === undefined) {
		return '';
	}
	if (Array.isArray(value)) {
		return `[${value.map((item) => formatJsonValue(item)).join(', ')}]`;
	}
	if (value && typeof value === 'object') {
		return JSON.stringify(value);
	}
	return String(value);
}

export function formatNumber(value: unknown): string {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		return 'n/a';
	}
	const abs = Math.abs(value);
	if (abs !== 0 && (abs < 0.001 || abs >= 10000)) {
		return value.toExponential(3);
	}
	return value.toLocaleString('en-US', {
		maximumFractionDigits: 6,
	});
}

export function loadRuns(): LoadedRun[] {
	if (!existsSync(traceRoot)) {
		return [];
	}

	return readdirSync(traceRoot)
		.filter((name) => name.endsWith('.json'))
		.sort()
		.flatMap((name) => {
			const path = resolve(traceRoot, name);
			const trace = readTrace(path, name);
			if (!trace) return [];
			return [
				{
					run_id: trace.run_id,
					path: `traces/${name}`,
					schema: trace.schema,
					run_name: trace.run_name,
					snapshot_count: trace.snapshots.length,
					completed_steps: trace.completed_steps,
					optimizer_classes: trace.optimizer_classes,
					full: trace,
				},
			];
		});
}

export const loadFingerprints = loadRuns;

export function countRunsByName(runs: LoadedRun[]): Map<string, number> {
	const counts = new Map<string, number>();
	for (const run of runs) {
		counts.set(run.run_name, (counts.get(run.run_name) ?? 0) + 1);
	}
	return counts;
}

export const countFingerprintsByOptimizer = countRunsByName;

export function runSnapshotCount(run: LoadedRun): number {
	return run.full.snapshots.length;
}

export const fingerprintSnapshotCount = runSnapshotCount;

export function runTaskId(): string {
	return 'nanogpt';
}

export const fingerprintTaskId = runTaskId;

export function isNanoGptTrace(value: unknown): value is NanoGptTrace {
	if (!value || typeof value !== 'object') {
		return false;
	}
	const candidate = value as Partial<NanoGptTrace>;
	return (
		candidate.schema === 'nanogpt_optimizer_trace' &&
		typeof candidate.run_name === 'string' &&
		typeof candidate.completed_steps === 'number' &&
		typeof candidate.snapshot_interval === 'number' &&
		Array.isArray(candidate.optimizer_classes) &&
		Array.isArray(candidate.metric_names) &&
		Array.isArray(candidate.snapshots)
	);
}

function readTrace(path: string, filename: string): NormalizedTrace | undefined {
	const parsed = JSON.parse(readFileSync(path, 'utf8')) as unknown;
	if (!isNanoGptTrace(parsed)) {
		return undefined;
	}
	return normalizeTrace(parsed, filename);
}

function normalizeTrace(trace: NanoGptTrace, filename: string): NormalizedTrace {
	const runId = basename(filename, '.json');
	const displayName = trace.run_name || runId;
	const parameterMetricNames = trace.metric_names.length
		? trace.metric_names
		: Array.from(new Set(trace.snapshots.flatMap((snapshot) => snapshot.parameters.flatMap((parameter) => Object.keys(parameter.metrics)))));

	return {
		...trace,
		run_id: runId,
		fingerprint_id: runId,
		display_name: displayName,
		optimizer: {
			name: displayName,
			family: trace.optimizer_classes.join(' + ') || 'optimizer',
			classes: trace.optimizer_classes,
		},
		parameter_metric_names: parameterMetricNames,
		snapshots: trace.snapshots.map((snapshot) => ({
			...snapshot,
			parameters: snapshot.parameters.map((parameter) => ({
				...parameter,
				ndim: parameter.ndim ?? parameter.shape.length,
				numel: parameter.numel ?? parameter.shape.reduce((product, size) => product * size, 1),
			})),
		})),
	};
}
