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
	schema_version: string;
	feature_schema_version?: string;
	world_id: string;
	optimizer: string;
	optimizer_family?: string;
	seed: number;
	feature_count: number;
	path: string;
}

export interface Fingerprint {
	fingerprint_id: string;
	schema_version: string;
	feature_schema_version?: string;
	feature_names: string[];
	features: Record<string, number>;
	raw_vector: number[];
	normalized_vector: number[];
	block_weights?: Record<string, number>;
	probe_config?: Record<string, JsonValue>;
	checkpoint_steps?: number[];
	checkpoint_records?: JsonValue[];
	path_length?: number;
	displacement?: number;
	world?: Record<string, JsonValue>;
	optimizer?: {
		name: string;
		family?: string;
		hparams?: Record<string, JsonValue>;
		param_groups?: Record<string, JsonValue>;
		metadata?: Record<string, JsonValue>;
		config_path?: string;
	};
}

export interface LoadedFingerprint extends FingerprintIndexEntry {
	full?: Fingerprint;
}

export interface FingerprintIndex {
	schema_version: string;
	fingerprint_root: string;
	fingerprints: FingerprintIndexEntry[];
}

const webRoot = process.cwd();
const repoRoot = resolve(webRoot, '..');

export const optimizerConfigs: OptimizerConfig[] = [
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
			schema_version: 'fingerprint_index_v1',
			fingerprint_root: 'fingerprints',
			fingerprints: [],
		};
	}
	return JSON.parse(readFileSync(path, 'utf8')) as FingerprintIndex;
}

export function loadFingerprints(): LoadedFingerprint[] {
	const index = loadFingerprintIndex();
	return index.fingerprints.map((entry) => ({
		...entry,
		full: readFingerprint(entry.path),
	}));
}

export function countFingerprintsByOptimizer(fingerprints: LoadedFingerprint[]): Map<string, number> {
	const counts = new Map<string, number>();
	for (const fingerprint of fingerprints) {
		counts.set(fingerprint.optimizer, (counts.get(fingerprint.optimizer) ?? 0) + 1);
	}
	return counts;
}

function readFingerprint(entryPath: string): Fingerprint | undefined {
	const candidates = [
		resolve(repoRoot, entryPath),
		resolve(webRoot, 'public', entryPath),
		resolve(webRoot, 'public', entryPath.replace(/^web\/public\//, '')),
	];
	for (const candidate of candidates) {
		if (existsSync(candidate)) {
			return JSON.parse(readFileSync(candidate, 'utf8')) as Fingerprint;
		}
	}
	return undefined;
}
