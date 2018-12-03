library(Seurat)
library(microbenchmark)

data_dir = "/path/to/data"

run_analysis <- function(data_dir) {
	# load data and create Seurat object
	sample.data <- Read10X(data_dir)
	sample <- CreateSeuratObject(raw.data = sample.data, min.cells = 3, min.genes = 200)

	# normalize data
	sample <- NormalizeData(object = sample, normalization.method = "LogNormalize", scale.factor = 1e4)

	# regress out total UMIs
	sample <- ScaleData(object = sample, vars.to.regress = c("nUMI"))

	# dimensionality reduction
	sample <- RunPCA(object = sample, pc.genes = sample@data@Dimnames[[1]], pcs.compute = 15, do.print = FALSE)
	#VizPCA(object = sample, pcs.use = 1:2)
	#PCAPlot(object = sample, dim.1 = 1, dim.2 = 2)

	# clustering
	sample <- FindClusters(object = sample, reduction.type = "pca", dims.use = 1:15, save.SNN = FALSE)
	#PrintFindClustersParams(object = sample)

	# tsne
	sample <- RunTSNE(object = sample, dims.use = 1:15, do.fast = TRUE)
	#TSNEPlot(object = sample, do.label = T, no.legend = T)
}

microbenchmark(run_analysis(data_dir), times=3)
