export interface TraceManifestEntry {
	id?: string;
	title: string;
	description: string;
	trace_url: string;
}

export interface TraceCatalog {
	traces: TraceManifestEntry[];
}

const traceCache = new Map<string, Promise<any>>();

export async function loadTraceManifest(manifestUrl: string): Promise<TraceManifestEntry[]> {
	return (await loadTraceCatalog(manifestUrl)).traces;
}

export async function loadTraceCatalog(manifestUrl: string): Promise<TraceCatalog> {
	const response = await fetch(manifestUrl);
	if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
	const value = await response.json();
	const entries = Array.isArray(value) ? value : (value as TraceCatalog)?.traces;
	if (!Array.isArray(entries)) throw new Error('Manifest must be an array or an object with a traces array');

	const traces = entries.map((entry, index) => {
		if (
			!entry ||
			typeof entry.title !== 'string' ||
			typeof entry.description !== 'string' ||
			typeof entry.trace_url !== 'string'
		) {
			throw new Error(`Invalid manifest entry at index ${index}`);
		}
		return {
			...entry,
			trace_url: new URL(entry.trace_url, response.url).href,
		};
	});
	return { traces };
}

export function loadTrace(entry: TraceManifestEntry): Promise<any> {
	const cached = traceCache.get(entry.trace_url);
	if (cached) return cached;

	const request = fetch(entry.trace_url)
		.then((response) => {
			if (!response.ok) throw new Error(`Trace request failed (${response.status})`);
			return response.json();
		})
		.then((value) => normalizeTrace(value, entry));
	traceCache.set(entry.trace_url, request);
	request.catch(() => traceCache.delete(entry.trace_url));
	return request;
}

function normalizeTrace(trace: any, entry: TraceManifestEntry): any {
	if (
		!trace ||
		trace.schema !== 'nanogpt_optimizer_trace' ||
		!Array.isArray(trace.snapshots) ||
		!Array.isArray(trace.optimizer_classes) ||
		!Array.isArray(trace.metric_names)
	) {
		throw new Error(`“${entry.title}” is not a NanoGPT optimizer trace`);
	}
	const runId = trace.run_id || entry.trace_url;
	const parameterMetricNames = trace.metric_names.length
		? trace.metric_names
		: Array.from(
				new Set(
					trace.snapshots.flatMap((snapshot: any) =>
						snapshot.parameters.flatMap((parameter: any) => Object.keys(parameter.metrics)),
					),
				),
			);
	return {
		...trace,
		run_id: runId,
		fingerprint_id: runId,
		display_name: entry.title,
		description: entry.description,
		optimizer: {
			name: entry.title,
			family: trace.optimizer_classes.join(' + ') || 'optimizer',
			classes: trace.optimizer_classes,
		},
		parameter_metric_names: parameterMetricNames,
		snapshots: trace.snapshots.map((snapshot: any) => ({
			...snapshot,
			parameters: snapshot.parameters.map((parameter: any) => ({
				...parameter,
				ndim: parameter.ndim ?? parameter.shape.length,
				numel: parameter.numel ?? parameter.shape.reduce((product: number, size: number) => product * size, 1),
			})),
		})),
	};
}
