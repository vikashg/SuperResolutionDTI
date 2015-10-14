ADD_EXECUTABLE(testMapping TestMapping.cpp)
TARGET_LINK_LIBRARIES(testMapping ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(IsolateProbTensors IsolateProbTensors.cpp)
TARGET_LINK_LIBRARIES(IsolateProbTensors ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(MaskImage_thres MaskImage_thres.cpp)
TARGET_LINK_LIBRARIES(MaskImage_thres ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(testSobol TestSobol.cpp)
TARGET_LINK_LIBRARIES(testSobol ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(extractValues ExtractValues.cpp)
TARGET_LINK_LIBRARIES(extractValues ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(imageDisplayProperties ImageDisplayProperties.cpp)
TARGET_LINK_LIBRARIES(imageDisplayProperties ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(EigenValues EigenValues.cpp)
TARGET_LINK_LIBRARIES(EigenValues ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(removeSlices RemoveSlices.cpp)
TARGET_LINK_LIBRARIES(removeSlices ${ITK_LIBRARIES} SuperResolution) 

#ADD_EXECUTABLE(propagateGradients PropagateGradients.cpp)
#TARGET_LINK_LIBRARIES(propagateGradients ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(ConvertLogExpTensorImageFilter ConvertLogExpTensorImageFilter.cpp)
TARGET_LINK_LIBRARIES(ConvertLogExpTensorImageFilter ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(tensorValues TensorValues.cpp)
TARGET_LINK_LIBRARIES(tensorValues ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(extractScalarImage ExtractScalarImageRegion.cpp)
TARGET_LINK_LIBRARIES(extractScalarImage ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(MaskInv MaskInv.cpp)
TARGET_LINK_LIBRARIES(MaskInv ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(extractTensorImage ExtractTensorImageRegion.cpp)
TARGET_LINK_LIBRARIES(extractTensorImage ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(extractVectorImage ExtractVectorImageRegion.cpp)
TARGET_LINK_LIBRARIES(extractVectorImage ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(ReplaceB0CutOff ReplaceB0CutOff.cpp)
TARGET_LINK_LIBRARIES(ReplaceB0CutOff ${ITK_LIBRARIES} SuperResolution)  

ADD_EXECUTABLE(ConvertTTK2FSL ConvertTTK2FSL.cpp)
TARGET_LINK_LIBRARIES(ConvertTTK2FSL ${ITK_LIBRARIES} SuperResolution)  

ADD_EXECUTABLE(PropagateGradients1 PropagateGradients1.cpp)
TARGET_LINK_LIBRARIES(PropagateGradients1 ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(TestWeightedNLS TestWeightedNLS.cpp)
TARGET_LINK_LIBRARIES(TestWeightedNLS ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(PropagateTransformGradients PropagateTransformGradients.cpp)
TARGET_LINK_LIBRARIES(PropagateTransformGradients ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(DeformationField_Transform DeformationField_Transform.cpp)
TARGET_LINK_LIBRARIES(DeformationField_Transform ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(StepWiseTransformGradients StepWiseTransformGradients.cpp)
TARGET_LINK_LIBRARIES(StepWiseTransformGradients ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(ComposeLinearNNonLinearTransform ComposeLinearNNonLinearTransform.cpp)
TARGET_LINK_LIBRARIES(ComposeLinearNNonLinearTransform ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(TransformPoints TransformPoints.cpp)
TARGET_LINK_LIBRARIES(TransformPoints ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(TestMappingDispField TestMappingDispField.cpp)
TARGET_LINK_LIBRARIES(TestMappingDispField ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(ComputeScalarValues ComputeScalarValues.cpp)
TARGET_LINK_LIBRARIES(ComputeScalarValues ${ITK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(TransformGradientTable_1 TransformGradientTable_1.cpp)
TARGET_LINK_LIBRARIES(TransformGradientTable_1 ${ITK_LIBRARIES} SuperResolution)
