import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface OptimizerConfig {
	name: string;
	description: string;
	hparams: Record<string, JsonValue>;
	paramGroups: Record<string, string>;
}

export interface FingerprintIndexEntry {
	fingerprint_id: string;
	schema: 'optimizer_fingerprint';
	task_id?: string;
	optimizer: string;
	optimizer_family?: string;
	seed: number;
	snapshot_count?: number;
	path: string;
}

export type MetricValue = number | null;

export interface SnapshotFingerprint {
	schema: 'optimizer_fingerprint';
	fingerprint_id: string;
	run_id: string;
	task: {
		id: string;
		dataset: string;
		batch_size: number;
		seed: number;
		max_steps: number;
		snapshot_interval: number;
		svd_max_dim: number;
	};
	model: {
		id: string;
	};
	optimizer: {
		name: string;
		family: string;
		[key: string]: JsonValue;
	};
	metric_names: string[];
	parameter_metric_names: string[];
	snapshots: {
		step: number;
		metrics: Record<string, MetricValue>;
		parameters: {
			name: string;
			shape: number[];
			ndim: number;
			numel: number;
			metrics: Record<string, MetricValue>;
		}[];
	}[];
}

export interface LoadedFingerprint extends FingerprintIndexEntry {
	full: SnapshotFingerprint;
}

export interface FingerprintIndex {
	schema: 'fingerprint_index';
	fingerprint_root: string;
	fingerprints: FingerprintIndexEntry[];
}

const webRoot = process.cwd();
const repoRoot = resolve(webRoot, '..');

export const optimizerConfigs: OptimizerConfig[] = [
	{
		name: 'adafactor',
		description: 'Adafactor factored coordinatewise updates over all trainable parameters.',
		hparams: {
			lr: 0.001,
			beta2_decay: -0.8,
			eps: [null, 0.001],
			d: 1.0,
			weight_decay: 0.01,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
	{
		name: 'adamw',
		description: 'AdamW over all trainable parameters.',
		hparams: {
			lr: 0.001,
			betas: [0.9, 0.95],
			weight_decay: 0.01,
			eps: 1.0e-8,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
	{
		name: 'lion',
		description: 'Lion sign update with momentum over all trainable parameters.',
		hparams: {
			lr: 0.0001,
			betas: [0.9, 0.99],
			weight_decay: 0.01,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
	{
		name: 'muon',
		description: 'Matrix-like parameters use Muon; auxiliary parameters use AdamW.',
		hparams: {
			lr: 0.02,
			weight_decay: 0.01,
			mu: 0.95,
			nesterov: true,
			ns_steps: 12,
			adam_lr: 0.001,
			adam_betas: [0.9, 0.95],
			adam_eps: 1.0e-8,
			adam_weight_decay: 0.01,
		},
		paramGroups: {
			matrix: 'ndim>=2',
			aux: 'ndim<2',
		},
	},
	{
		name: 'sam_adamw',
		description: 'SAM perturbation step with AdamW base optimizer over all trainable parameters.',
		hparams: {
			lr: 0.001,
			betas: [0.9, 0.95],
			weight_decay: 0.01,
			eps: 1.0e-8,
			rho: 0.05,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
	{
		name: 'sgd',
		description: 'Plain SGD over all trainable parameters.',
		hparams: {
			lr: 0.1,
			weight_decay: 0.0,
			momentum: 0.0,
			dampening: 0.0,
			nesterov: false,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
	{
		name: 'sgd_momentum',
		description: 'SGD with momentum over all trainable parameters.',
		hparams: {
			lr: 0.1,
			weight_decay: 0.0,
			momentum: 0.9,
			dampening: 0.0,
			nesterov: false,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
	{
		name: 'shampoo_default',
		description: 'Distributed Shampoo over matrix-like parameters with AdamW auxiliary parameters.',
		hparams: {
			lr: 0.01,
			betas: [0.9, 0.9],
			beta2: 0.9,
			epsilon: 1.0e-12,
			weight_decay: 0.01,
			adam_lr: 0.001,
			adam_betas: [0.9, 0.95],
			preconditioner: 'default',
			max_preconditioner_dim: 8192,
			precondition_frequency: 1,
			start_preconditioning_step: -1,
			grafting_epsilon: 1.0e-15,
		},
		paramGroups: {
			matrix: 'ndim>=2',
			aux: 'ndim<2',
		},
	},
	{
		name: 'shampoo_pinv_one_sided',
		description: 'Pseudoinverse-root one-sided Shampoo over matrix-like parameters with AdamW auxiliary parameters.',
		hparams: {
			lr: 0.01,
			betas: [0.9, 0.9],
			beta2: 0.9,
			epsilon: 0.0,
			weight_decay: 0.01,
			adam_lr: 0.001,
			adam_betas: [0.9, 0.95],
			preconditioner: 'pinv_one_sided',
			max_preconditioner_dim: 8192,
			precondition_frequency: 1,
			start_preconditioning_step: -1,
			grafting_epsilon: 1.0e-15,
		},
		paramGroups: {
			matrix: 'ndim>=2',
			aux: 'ndim<2',
		},
	},
	{
		name: 'soap',
		description:
			'SOAP-style Adam updates in the Shampoo preconditioner eigenbasis for matrix-like parameters with AdamW auxiliary parameters.',
		hparams: {
			lr: 0.001,
			betas: [0.9, 0.95],
			beta2: 0.95,
			epsilon: 1.0e-8,
			weight_decay: 0.01,
			adam_lr: 0.001,
			adam_betas: [0.9, 0.95],
			preconditioner: 'soap',
			max_preconditioner_dim: 8192,
			precondition_frequency: 1,
			start_preconditioning_step: -1,
			grafting_epsilon: 1.0e-8,
		},
		paramGroups: {
			matrix: 'ndim>=2',
			aux: 'ndim<2',
		},
	},
	{
		name: 'sophia_g',
		description: 'Sophia-G style clipped diagonal curvature updates over all trainable parameters.',
		hparams: {
			lr: 0.001,
			betas: [0.965, 0.99],
			rho: 0.04,
			weight_decay: 0.01,
			eps: 1.0e-12,
		},
		paramGroups: {
			all: 'trainable',
		},
	},
];

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

export function loadFingerprintIndex(): FingerprintIndex {
	const path = resolve(webRoot, 'public', 'fingerprints.json');
	if (!existsSync(path)) {
		return {
			schema: 'fingerprint_index',
			fingerprint_root: 'fingerprints',
			fingerprints: [],
		};
	}
	return JSON.parse(readFileSync(path, 'utf8')) as FingerprintIndex;
}

export function loadFingerprints(): LoadedFingerprint[] {
	const index = loadFingerprintIndex();
	return index.fingerprints.flatMap((entry) => {
		const full = readFingerprint(entry.path);
		return full ? [{ ...entry, full }] : [];
	});
}

export function countFingerprintsByOptimizer(fingerprints: LoadedFingerprint[]): Map<string, number> {
	const counts = new Map<string, number>();
	for (const fingerprint of fingerprints) {
		counts.set(fingerprint.optimizer, (counts.get(fingerprint.optimizer) ?? 0) + 1);
	}
	return counts;
}

export function isSnapshotFingerprint(value: unknown): value is SnapshotFingerprint {
	if (!value || typeof value !== 'object') {
		return false;
	}
	const candidate = value as Partial<SnapshotFingerprint>;
	return (
		candidate.schema === 'optimizer_fingerprint' &&
		typeof candidate.fingerprint_id === 'string' &&
		typeof candidate.run_id === 'string' &&
		typeof candidate.task === 'object' &&
		typeof candidate.model === 'object' &&
		typeof candidate.optimizer === 'object' &&
		Array.isArray(candidate.metric_names) &&
		Array.isArray(candidate.parameter_metric_names) &&
		Array.isArray(candidate.snapshots)
	);
}

export function fingerprintTaskId(fingerprint: LoadedFingerprint): string {
	return fingerprint.full.task.id;
}

export function fingerprintSnapshotCount(fingerprint: LoadedFingerprint): number | undefined {
	return fingerprint.full.snapshots.length;
}

function readFingerprint(entryPath: string): SnapshotFingerprint | undefined {
	const candidates = [
		resolve(repoRoot, entryPath),
		resolve(webRoot, 'public', entryPath),
		resolve(webRoot, 'public', entryPath.replace(/^web\/public\//, '')),
	];
	for (const candidate of candidates) {
		if (existsSync(candidate)) {
			const fingerprint = JSON.parse(readFileSync(candidate, 'utf8')) as unknown;
			return isSnapshotFingerprint(fingerprint) ? fingerprint : undefined;
		}
	}
	return undefined;
}
