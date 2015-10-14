ADD_LIBRARY(SuperResolution sobol.cpp
			    MapFilterLR2HR.cpp
			    ComposeImage.cpp
			    UnweightedLeastSquaresTensorFit.cpp
			    TensorUtilites.cpp
				JointTensorEstimation_LNLS.cpp
				WeightedLeastSquares_CWLS.cpp
			    CopyImage.cpp
				ComputeSigma_LR.cpp
				TotalEnergy.cpp
				JacobianComputation.cpp
				MapFilterLR2HRDispField.cpp
			   WeightedLeastSquares_new.cpp
				TransformGradients.cpp)
