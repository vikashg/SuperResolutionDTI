ADD_LIBRARY(SuperResolution sobol.cpp
			    MapFilterLR2HR.cpp
			    ComposeImage.cpp
			    UnweightedLeastSquaresTensorFit.cpp
			    TensorUtilites.cpp
			    JointTensorEstimation.cpp
			    CopyImage.cpp
				ComputeSigma_LR.cpp
				ComputeSigma.cpp
				TotalEnergy.cpp
				JacobianComputation.cpp
				MapFilterLR2HRDispField.cpp
				TransformGradients.cpp)
