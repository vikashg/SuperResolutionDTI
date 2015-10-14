#include "itkImage.h"
#include "TensorUtilites.h"
#include "WeightedLeastSquares.h"
#include "itkDiffusionTensor3D.h"

int main()
{
	WeightedLeastSquares wls;
	typedef float RealType;
 	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;	

	DiffusionTensorType D;
	D.Fill(1);

	std::cout <<D << std::endl;
	wls.ComputeJacobian(D);

	return 0;
}
