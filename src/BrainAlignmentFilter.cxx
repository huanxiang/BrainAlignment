/*==========================================================================================
 *
 *	Program:	TOMI
 *	Module:		BrainAlignment
 *	Language: C++
 *	Data:			$Date: 2014-03-17 10:47:53$
 *	Version:	$Revision: 0.1$
 *
 *	Copyright Philips Research China. All rights reserved.
 *
 *  This program aligns the brain image into the mni atlas space using affine registration.
 *
 *=========================================================================================*/

#ifndef _BrainAlignmentFilter_cxx_
#define _BrainAlignmentFilter_cxx_

#include "BrainAlignmentFilter.h"

#include <itkCenteredVersorTransformInitializer.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include "itkVersorRigid3DTransformOptimizer.h"
#include "itkImageRegistrationMethod.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkMultiResolutionPyramidImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkTimeProbe.h"
#include "itkScaleSkewVersor3DTransform.h"
#include "itkImageMaskSpatialObject.h"

#include "itkDualResampleImageFilter.h"


template<class TImage>
BrainAlignment<TImage>::BrainAlignment()
{
	m_Atlas = 0;
	m_InputImage = 0;
	m_RigidOutputImage = 0;
	m_AffineOutputImage = 0;
	m_AffineCSF = 0;
	m_AffineGM = 0;
	m_AffineWM = 0;
	m_AffineLabel = 0;

	m_NumberOfSamples = 100000;
	m_DownsampleFactor = 2;
}

template<class TImage>
BrainAlignment<TImage>::~BrainAlignment()
{

}

template<class TImage>
typename BrainAlignment<TImage>::ImagePointerType	BrainAlignment<TImage>::DownsampleImage(ImagePointerType image)
{
	typedef itk::ResampleImageFilter< ImageType, ImageType >	DownsampleFilterType;
  typedef itk::LinearInterpolateImageFunction< ImageType, double >    InterpolatorType;
	typedef itk::IdentityTransform<double>	TransformType;
	TransformType::Pointer idTransform = TransformType::New();

	DownsampleFilterType::Pointer downsampler = DownsampleFilterType::New();
  InterpolatorType::Pointer downsampleInterpolator = InterpolatorType::New();

	ImageType::SizeType inputSize = image->GetBufferedRegion().GetSize();
	ImageType::SizeType downsampleSize;
	downsampleSize[0] = inputSize[0] / m_DownsampleFactor;
	downsampleSize[1] = inputSize[1] / m_DownsampleFactor;
	downsampleSize[2] = inputSize[2] / m_DownsampleFactor;

	ImageType::SpacingType inputSpacing = image->GetSpacing();
	ImageType::SpacingType downsampleSpacing;
	downsampleSpacing[0] = inputSpacing[0] / (float)downsampleSize[0] * (float)inputSize[0];
	downsampleSpacing[1] = inputSpacing[1] / (float)downsampleSize[1] * (float)inputSize[1];
	downsampleSpacing[2] = inputSpacing[2] / (float)downsampleSize[2] * (float)inputSize[2];

	downsampler->SetTransform(idTransform);
	downsampler->SetInput( image );
	downsampler->SetInterpolator( downsampleInterpolator );

	downsampler->SetSize(    downsampleSize );
	downsampler->SetOutputOrigin(  image->GetOrigin() );
	downsampler->SetOutputSpacing( downsampleSpacing );
	downsampler->SetOutputDirection( image->GetDirection() );
	downsampler->SetDefaultPixelValue( 0 );
	downsampler->Update();
	
	ImageType::Pointer downsampleImage = downsampler->GetOutput();

	return downsampleImage;
	
}

template<class TImage>
void BrainAlignment<TImage>::Update()
{
	// Typical type definitions of registration components
	const unsigned int Dimension = 3;

  typedef itk::VersorRigid3DTransformOptimizer           RigidOptimizerType;

	typedef itk::RegularStepGradientDescentOptimizer       AffineOptimizerType;

  typedef itk::MattesMutualInformationImageToImageMetric< 
                                    InternalImageType, 
                                    InternalImageType >    MetricType;
  

  typedef itk::LinearInterpolateImageFunction< 
                                    InternalImageType,
                                    double          >    InterpolatorType;

  typedef itk::MultiResolutionImageRegistrationMethod< 
                                    InternalImageType, 
                                    InternalImageType >   RegistrationType;

  typedef itk::MultiResolutionPyramidImageFilter<
                                    InternalImageType,
                                    InternalImageType >   FixedImagePyramidType;
  typedef itk::MultiResolutionPyramidImageFilter<
                                    InternalImageType,
                                    InternalImageType >   MovingImagePyramidType;

  typedef itk::CastImageFilter< 
                        ImageType, InternalImageType > FixedCastFilterType;
  typedef itk::CastImageFilter< 
                        ImageType, InternalImageType > MovingCastFilterType;

  typedef itk::CenteredTransformInitializer< RigidTransformType,
                                             ImageType, 
                                             ImageType 
                                                 >  TransformInitializerType;
	typedef itk::ResampleImageFilter< 
                            ImageType, 
                            ImageType >    ResampleFilterType;

  typedef itk::LinearInterpolateImageFunction< 
                                 ImageType, double >  ResampleInterpolatorType;

	typedef itk::ImageFileReader<ImageType>	ReaderType;
	
	typedef itk::ImageFileWriter<ImageType>	WriterType;

	typedef itk::ImageFileReader<InternalImageType>	FloatReaderType;

	typedef itk::ResampleImageFilter< 
                            InternalImageType, 
                            InternalImageType >    FloatResampleFilterType;

  typedef itk::LinearInterpolateImageFunction< 
                                 InternalImageType, double >  FloatResampleInterpolatorType;

	// Load the atlas image
	ReaderType::Pointer atlasReader = ReaderType::New();
	std::string atlasName = m_AtlasPath + "/mni_icbm152_t2_tal_nlin_sym_09c.nii";
	atlasReader->SetFileName(atlasName);
	try
	{
		atlasReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}


	m_Atlas = atlasReader->GetOutput();

	ImageType::Pointer downsampledAtlas = DownsampleImage(m_Atlas);

	//ImageType::Pointer downsampledInputImage = DownsampleImage(m_InputImage);


	/* First perform rigid registration to align subject to atlas*/
  MetricType::Pointer         rigidMetric        = MetricType::New();
  RigidOptimizerType::Pointer rigidOptimizer     = RigidOptimizerType::New();
  InterpolatorType::Pointer   rigidInterpolator  = InterpolatorType::New();
  RegistrationType::Pointer   rigidRegistration  = RegistrationType::New();
  
  FixedImagePyramidType::Pointer fixedRigidImagePyramid = 
      FixedImagePyramidType::New();
  MovingImagePyramidType::Pointer movingRigidImagePyramid =
      MovingImagePyramidType::New();

	// Key components of the rigid registration
  rigidRegistration->SetMetric(        rigidMetric        );
  rigidRegistration->SetOptimizer(     rigidOptimizer     );
  rigidRegistration->SetInterpolator(  rigidInterpolator  );
  rigidRegistration->SetFixedImagePyramid( fixedRigidImagePyramid );
  rigidRegistration->SetMovingImagePyramid( movingRigidImagePyramid ); 

  RigidTransformType::Pointer  rigidTransform = RigidTransformType::New();
  rigidRegistration->SetTransform( rigidTransform );

	// Set parameter for the MI metric
	const int numberOfPixels = m_Atlas->GetBufferedRegion().GetNumberOfPixels();
  rigidMetric->SetNumberOfHistogramBins(50);
  rigidMetric->SetNumberOfSpatialSamples(m_NumberOfSamples);


	// Cast the short image type to float image type
  FixedCastFilterType::Pointer fixedRigidCaster   = FixedCastFilterType::New();
  MovingCastFilterType::Pointer movingRigidCaster = MovingCastFilterType::New();

	fixedRigidCaster->SetInput(  downsampledAtlas );
	movingRigidCaster->SetInput( m_InputImage );

  rigidRegistration->SetFixedImage(    fixedRigidCaster->GetOutput()    );
  rigidRegistration->SetMovingImage(   movingRigidCaster->GetOutput()   );

  fixedRigidCaster->Update();

  rigidRegistration->SetFixedImageRegion( 
       fixedRigidCaster->GetOutput()->GetBufferedRegion() );	

	// Initialize the transform
  //TransformInitializerType::Pointer initializer = 
  //                                        TransformInitializerType::New();

  //initializer->SetTransform(   rigidTransform );
  //initializer->SetFixedImage(  m_Atlas );
  //initializer->SetMovingImage( m_InputImage );
  //initializer->GeometryOn();

  //typedef RigidTransformType::VersorType  VersorType;
  //typedef VersorType::VectorType     VectorType;

  //VersorType     rotation;
  //VectorType     axis;
  //
  //axis[0] = 1;
  //axis[1] = 0;
  //axis[2] = 0;
  //const double angle = 0;
  //rotation.Set(  axis, angle  );
  //rigidTransform->SetRotation( rotation );
  //rigidTransform->SetTranslation( translation );
  //initializer->InitializeTransform();

  rigidRegistration->SetInitialTransformParameters( rigidTransform->GetParameters() );

	// Set optimizer scales and parameters
  typedef RigidOptimizerType::ScalesType       RigidOptimizerScalesType;
  RigidOptimizerScalesType rigidOptimizerScales( rigidTransform->GetNumberOfParameters() );

  rigidOptimizerScales[0] = 1000;
  rigidOptimizerScales[1] = 1000;
  rigidOptimizerScales[2] = 1000;
  rigidOptimizerScales[3] = 1;
  rigidOptimizerScales[4] = 1;
  rigidOptimizerScales[5] = 1;

  rigidOptimizer->SetScales( rigidOptimizerScales );

  rigidOptimizer->SetMaximumStepLength( 2  ); 
  rigidOptimizer->SetMinimumStepLength( 0.005 );

  rigidOptimizer->SetNumberOfIterations( 1500 );
  rigidOptimizer->MinimizeOn();

  rigidRegistration->SetNumberOfLevels( 3 );

	// Start registration optimization
	std::cout<<"Start rigid registration..."<<std::endl;
	itk::TimeProbe clock;
	clock.Start();
  try 
  { 
    rigidRegistration->Update(); 
		std::cout << "Optimizer stop condition: "
          << rigidRegistration->GetOptimizer()->GetStopConditionDescription()
          << std::endl;

  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    exit;
  } 
	clock.Stop();
	std::cout << "Time: " << clock.GetTotal() << std::endl;
  
  RigidOptimizerType::ParametersType rigidParameters = 
                    rigidRegistration->GetLastTransformParameters();

	m_RigidTransform = RigidTransformType::New();
	m_RigidTransform->SetParameters(rigidParameters);
	m_RigidTransform->SetCenter(rigidTransform->GetCenter());

	// Resample the subject image using the resulting rigid transform using downsampled resolution
	ResampleFilterType::Pointer rigidResampler = ResampleFilterType::New();
  ResampleInterpolatorType::Pointer rigidResampleInterpolator = ResampleInterpolatorType::New();

	rigidResampler->SetTransform( m_RigidTransform );
	rigidResampler->SetInput( m_InputImage );
	rigidResampler->SetInterpolator( rigidResampleInterpolator );

	rigidResampler->SetSize(    downsampledAtlas->GetLargestPossibleRegion().GetSize() );
	rigidResampler->SetOutputOrigin(  downsampledAtlas->GetOrigin() );
	rigidResampler->SetOutputSpacing( downsampledAtlas->GetSpacing() );
	rigidResampler->SetOutputDirection( downsampledAtlas->GetDirection() );
	rigidResampler->SetDefaultPixelValue( 0 );
	rigidResampler->Update();

	m_RigidOutputImage = rigidResampler->GetOutput();

	//// write output for debug
	//typedef itk::ImageFileWriter<ImageType>	WriterType;
	//WriterType::Pointer rigidWriter = WriterType::New();
	//rigidWriter->SetInput(m_RigidOutputImage);
	//rigidWriter->SetFileName("rigidOutput.nii");
	//try
	//{
	//	rigidWriter->Update();
	//}
	//catch(itk::ExceptionObject & err)
	//{
	//	std::cerr<< err << std::endl;
	//}


	/* Second perform affine registration to align atlas to template */


	// Key components of the rigid registration
	MetricType::Pointer						affineMetric        = MetricType::New();
  AffineOptimizerType::Pointer  affineOptimizer     = AffineOptimizerType::New();
  InterpolatorType::Pointer			affineInterpolator  = InterpolatorType::New();
  RegistrationType::Pointer			affineRegistration  = RegistrationType::New();
	AffineTransformType::Pointer  affineTransform			= AffineTransformType::New();

  affineRegistration->SetMetric(        affineMetric        );
  affineRegistration->SetOptimizer(     affineOptimizer     );
  affineRegistration->SetInterpolator(  affineInterpolator  );
  affineRegistration->SetTransform(			affineTransform			);

  affineRegistration->SetInitialTransformParameters( affineTransform->GetParameters() );

	// Set metric parameters
  affineMetric->SetNumberOfHistogramBins(50);
  affineMetric->SetNumberOfSpatialSamples(m_NumberOfSamples);

	// Set optimizer scales and parameters
  typedef AffineOptimizerType::ScalesType						AffineOptimizerScalesType;
  AffineOptimizerScalesType affineOptimizerScales(	affineTransform->GetNumberOfParameters() );
  affineOptimizerScales[0] =  1.0;
  affineOptimizerScales[1] =  1.0;
  affineOptimizerScales[2] =  1.0;
  affineOptimizerScales[3] =  1.0;
  affineOptimizerScales[4] =  1.0;
  affineOptimizerScales[5] =  1.0;
  affineOptimizerScales[6] =  1.0;
  affineOptimizerScales[7] =  1.0;
  affineOptimizerScales[8] =  1.0;
  affineOptimizerScales[9]  =  0.001;
  affineOptimizerScales[10] =  0.001;
  affineOptimizerScales[11] =  0.001;
  affineOptimizer->SetScales( affineOptimizerScales );

  affineOptimizer->SetMaximumStepLength( 1 );
  affineOptimizer->SetMinimumStepLength( 0.005 );
  affineOptimizer->SetNumberOfIterations( 200 );

	affineOptimizer->MinimizeOn();

	// Cast the short image type to float image type
  FixedCastFilterType::Pointer	fixedAffineCaster  = FixedCastFilterType::New();
  MovingCastFilterType::Pointer movingAffineCaster = MovingCastFilterType::New();

	fixedAffineCaster->SetInput(  m_RigidOutputImage );
	movingAffineCaster->SetInput( downsampledAtlas );

  affineRegistration->SetFixedImage(    fixedAffineCaster->GetOutput()    );
  affineRegistration->SetMovingImage(   movingAffineCaster->GetOutput()   );

  fixedAffineCaster->Update();

  affineRegistration->SetFixedImageRegion( 
       fixedAffineCaster->GetOutput()->GetBufferedRegion() );	

	// Start affine registration
	std::cout<<"Start Affine Registration..."<<std::endl;
	itk::TimeProbe clock2;
	clock2.Start();
  try
  {
    affineRegistration->Update();
    std::cout << "Optimizer stop condition: "
              << affineRegistration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
  }
	clock2.Stop();
	std::cout << "Time: " << clock2.GetTotal() << std::endl;

	AffineOptimizerType::ParametersType affineParameters = 
                  affineRegistration->GetLastTransformParameters();


	m_AffineTransform = AffineTransformType::New();
	m_AffineTransform->SetParameters(affineParameters);
	m_AffineTransform->SetCenter(affineTransform->GetCenter());

	// Resample the atlas image using the resulting affine transform
	ResampleFilterType::Pointer affineResampler = ResampleFilterType::New();
  ResampleInterpolatorType::Pointer affineResampleInterpolator = ResampleInterpolatorType::New();

	affineResampler->SetTransform( m_AffineTransform );
	affineResampler->SetInput( m_Atlas );
	affineResampler->SetInterpolator( affineResampleInterpolator );

	affineResampler->SetSize(    m_Atlas->GetLargestPossibleRegion().GetSize() );
	affineResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	affineResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	affineResampler->SetOutputDirection( m_Atlas->GetDirection() );
	affineResampler->SetDefaultPixelValue( 0 );
	affineResampler->Update();

	m_AffineOutputImage = affineResampler->GetOutput();

	// Resample the subject image using the resulting rigid transform using original resolution
	ResampleFilterType::Pointer hrRigidResampler = ResampleFilterType::New();

	hrRigidResampler->SetTransform( m_RigidTransform );
	hrRigidResampler->SetInput( m_InputImage );
	hrRigidResampler->SetInterpolator( rigidResampleInterpolator );

	hrRigidResampler->SetSize(    m_Atlas->GetLargestPossibleRegion().GetSize() );
	hrRigidResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	hrRigidResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	hrRigidResampler->SetOutputDirection( m_Atlas->GetDirection() );
	hrRigidResampler->SetDefaultPixelValue( 0 );
	hrRigidResampler->Update();

	m_RigidOutputImage = hrRigidResampler->GetOutput();

	// Resample the atlas mask using the resulting affine transform
	ReaderType::Pointer maskReader = ReaderType::New();
	std::string maskName = m_AtlasPath + "/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii";
	maskReader->SetFileName(maskName);
	try
	{
		maskReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	ResampleFilterType::Pointer maskResampler = ResampleFilterType::New();
	maskResampler->SetTransform( m_AffineTransform );
	maskResampler->SetInput( maskReader->GetOutput() );
	maskResampler->SetInterpolator( affineResampleInterpolator );

	maskResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	maskResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	maskResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	maskResampler->SetOutputDirection( m_Atlas->GetDirection() );
	maskResampler->SetDefaultPixelValue( 0 );
	maskResampler->Update();

	m_AffineMask = maskResampler->GetOutput();

	// Resample the atlas WM using the resulting affine transform
	typedef itk::LinearInterpolateImageFunction<InternalImageType>	FloatInterpolatorType;
	FloatInterpolatorType::Pointer floatInterpolator = FloatInterpolatorType::New();

	FloatReaderType::Pointer wmReader = FloatReaderType::New();
	std::string wmName = m_AtlasPath + "/mni_icbm152_wm_tal_nlin_sym_09c.nii";
	wmReader->SetFileName(wmName);
	try
	{
		wmReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer wmResampler = FloatResampleFilterType::New();
	wmResampler->SetTransform( m_AffineTransform );
	wmResampler->SetInput( wmReader->GetOutput() );
	wmResampler->SetInterpolator( floatInterpolator );

	wmResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	wmResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	wmResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	wmResampler->SetOutputDirection( m_Atlas->GetDirection() );
	wmResampler->SetDefaultPixelValue( 0 );
	wmResampler->Update();

	m_AffineWM = wmResampler->GetOutput();

	// Resample the atlas WM using the resulting affine transform
	FloatReaderType::Pointer gmReader = FloatReaderType::New();
	std::string gmName = m_AtlasPath + "/mni_icbm152_gm_tal_nlin_sym_09c.nii";
	gmReader->SetFileName(gmName);
	try
	{
		gmReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer gmResampler = FloatResampleFilterType::New();
	gmResampler->SetTransform( m_AffineTransform );
	gmResampler->SetInput( gmReader->GetOutput() );
	gmResampler->SetInterpolator( floatInterpolator );

	gmResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	gmResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	gmResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	gmResampler->SetOutputDirection( m_Atlas->GetDirection() );
	gmResampler->SetDefaultPixelValue( 0 );
	gmResampler->Update();

	m_AffineGM = gmResampler->GetOutput();

	// Resample the atlas CSF using the resulting affine transform
	FloatReaderType::Pointer csfReader = FloatReaderType::New();
	std::string csfName = m_AtlasPath + "/mni_icbm152_csf_tal_nlin_sym_09c.nii";
	csfReader->SetFileName(csfName);
	try
	{
		csfReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer csfResampler = FloatResampleFilterType::New();
	csfResampler->SetTransform( m_AffineTransform );
	csfResampler->SetInput( csfReader->GetOutput() );
	csfResampler->SetInterpolator( floatInterpolator );

	csfResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	csfResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	csfResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	csfResampler->SetOutputDirection( m_Atlas->GetDirection() );
	csfResampler->SetDefaultPixelValue( 0 );
	csfResampler->Update();

	m_AffineCSF = csfResampler->GetOutput();

	// Resample the label using the resulting affine transform
	FloatReaderType::Pointer labelReader = FloatReaderType::New();
	std::string labelName = m_AtlasPath + "/mni_label.nii";
	labelReader->SetFileName(csfName);
	try
	{
		labelReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer labelResampler = FloatResampleFilterType::New();
	labelResampler->SetTransform( m_AffineTransform );
	labelResampler->SetInput( labelReader->GetOutput() );
	labelResampler->SetInterpolator( floatInterpolator );

	labelResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	labelResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	labelResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	labelResampler->SetOutputDirection( m_Atlas->GetDirection() );
	labelResampler->SetDefaultPixelValue( 0 );
	labelResampler->Update();

	m_AffineLabel = labelResampler->GetOutput();

}


template<class TImage>
void BrainAlignment<TImage>::UpdateRSS()
{
	const unsigned int Dimension = 3;
	typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;

  typedef itk::MattesMutualInformationImageToImageMetric< 
                                    InternalImageType, 
                                    InternalImageType >    MetricType;
  
  typedef itk::LinearInterpolateImageFunction< 
                                    InternalImageType,
                                    double          >    InterpolatorType;

  typedef itk::MultiResolutionImageRegistrationMethod< 
                                    InternalImageType, 
                                    InternalImageType >   RegistrationType;

	typedef itk::ScaleSkewVersor3DTransform<double>					TransformType;

  typedef itk::MultiResolutionPyramidImageFilter<
                                    InternalImageType,
                                    InternalImageType >   FixedImagePyramidType;
  typedef itk::MultiResolutionPyramidImageFilter<
                                    InternalImageType,
                                    InternalImageType >   MovingImagePyramidType;

  typedef itk::CastImageFilter< 
                        ImageType, InternalImageType > FixedCastFilterType;
  typedef itk::CastImageFilter< 
                        ImageType, InternalImageType > MovingCastFilterType;

	typedef itk::ImageFileReader<ImageType>	ReaderType;

	typedef itk::ResampleImageFilter< 
                            ImageType, 
                            ImageType >    ResampleFilterType;

  typedef itk::LinearInterpolateImageFunction< 
                                 ImageType, double >  ResampleInterpolatorType;

	typedef itk::ImageFileReader<InternalImageType>	FloatReaderType;

	typedef itk::DualResampleImageFilter< 
                            InternalImageType, 
                            InternalImageType >    FloatResampleFilterType;

  typedef itk::LinearInterpolateImageFunction< 
                                 InternalImageType, double >  FloatResampleInterpolatorType;

	typedef itk::DualResampleImageFilter<ImageType, ImageType>	DualResampleFilterType;


	// Load the atlas image
	ReaderType::Pointer atlasReader = ReaderType::New();
	std::string atlasName = m_AtlasPath + "/mni_icbm152_t2_tal_nlin_sym_09c.nii";
	atlasReader->SetFileName(atlasName);
	try
	{
		atlasReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	m_Atlas = atlasReader->GetOutput();

	// Downsample the atlas image
	ImageType::Pointer downsampledAtlas = DownsampleImage(m_Atlas);

  MetricType::Pointer         metric        = MetricType::New();
  OptimizerType::Pointer			optimizer     = OptimizerType::New();
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  RegistrationType::Pointer   registration  = RegistrationType::New();

  FixedImagePyramidType::Pointer fixedRigidImagePyramid = 
      FixedImagePyramidType::New();
  MovingImagePyramidType::Pointer movingRigidImagePyramid =
      MovingImagePyramidType::New();

	// Key components of the rigid registration
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetInterpolator(  interpolator  );
  registration->SetFixedImagePyramid( fixedRigidImagePyramid );
  registration->SetMovingImagePyramid( movingRigidImagePyramid ); 

  TransformType::Pointer  transform = TransformType::New();
  registration->SetTransform( transform );

	// Set parameter for the MI metric
	const int numberOfPixels = m_Atlas->GetBufferedRegion().GetNumberOfPixels();
  metric->SetNumberOfHistogramBins(50);
  metric->SetNumberOfSpatialSamples(m_NumberOfSamples);

	// Load the brain mask
	ReaderType::Pointer maskReader = ReaderType::New();
	std::string maskName = m_AtlasPath + "/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii";
	maskReader->SetFileName(maskName);
	try
	{
		maskReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	//typedef itk::Image<unsigned char, 3>	CharImageType;
	//typedef itk::CastImageFilter< 
	//										ImageType, CharImageType > CharCastFilterType;
	//CharCastFilterType::Pointer charCastFilter = CharCastFilterType::New();
	//charCastFilter->SetInput(maskReader->GetOutput());
	//charCastFilter->Update();

	//typedef itk::ImageMaskSpatialObject<Dimension> ImageMaskSpatialObject;
	//ImageMaskSpatialObject::Pointer maskSO = ImageMaskSpatialObject::New();
	//maskSO->SetImage(charCastFilter->GetOutput());
	//maskSO->Update();

	//metric->SetFixedImageMask(maskSO);

	// Cast the short image type to float image type
  FixedCastFilterType::Pointer fixedCaster   = FixedCastFilterType::New();
  MovingCastFilterType::Pointer movingCaster = MovingCastFilterType::New();

	fixedCaster->SetInput(  downsampledAtlas );
	movingCaster->SetInput( m_InputImage );

  registration->SetFixedImage(    fixedCaster->GetOutput()    );
  registration->SetMovingImage(   movingCaster->GetOutput()   );

  fixedCaster->Update();

  registration->SetFixedImageRegion( 
       fixedCaster->GetOutput()->GetBufferedRegion() );	

  registration->SetInitialTransformParameters( transform->GetParameters() );

  typedef OptimizerType::ScalesType		OptimizerScalesType;
  OptimizerScalesType optimizerScales(	transform->GetNumberOfParameters() );
  optimizerScales[0] =  1.0; //Versor X scale
  optimizerScales[1] =  1.0; //Versor Y scale
  optimizerScales[2] =  1.0; //Versor Z scale
  optimizerScales[3] =  0.001; //Translation X scale
  optimizerScales[4] =  0.001; //Translation Y scale
  optimizerScales[5] =  0.001; //Translation Z scale
  optimizerScales[6] =  10.0;  //Scale X scale
  optimizerScales[7] =  10.0;  //Scale Y scale
  optimizerScales[8] =  10.0;  //Scale Z scale
  optimizerScales[9] =  10.0;  // [9] to [14] are 6 skew scales
  optimizerScales[10] =  1000.0;
  optimizerScales[11] =  1000.0;
  optimizerScales[12] =  1000.0;
  optimizerScales[13] =  1000.0;
  optimizerScales[14] =  1000.0;
  optimizer->SetScales( optimizerScales );

  optimizer->SetMaximumStepLength( 1 );
  optimizer->SetMinimumStepLength( 0.01 );
  optimizer->SetNumberOfIterations( 1200 );

	optimizer->MinimizeOn();

	// Start RSS registration
	std::cout<<"Start RigidScaleSkew Registration..."<<std::endl;
	itk::TimeProbe clock2;
	clock2.Start();
  try
  {
    registration->Update();
    std::cout << "Optimizer stop condition: "
              << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
  }
	clock2.Stop();
	std::cout << "Time: " << clock2.GetTotal() << std::endl;

	OptimizerType::ParametersType parameters = 
                  registration->GetLastTransformParameters();


	m_RSSTransform = TransformType::New();
	m_RSSTransform->SetParameters(parameters);
	m_RSSTransform->SetCenter(transform->GetCenter());

	// Resample the atlas image using the resulting affine transform
	ResampleFilterType::Pointer rssResampler = ResampleFilterType::New();
  ResampleInterpolatorType::Pointer rssResampleInterpolator = ResampleInterpolatorType::New();

	rssResampler->SetTransform( m_RSSTransform );
	rssResampler->SetInput( m_InputImage );
	rssResampler->SetInterpolator( rssResampleInterpolator );

	rssResampler->SetSize(    m_Atlas->GetLargestPossibleRegion().GetSize() );
	rssResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	rssResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	rssResampler->SetOutputDirection( m_Atlas->GetDirection() );
	rssResampler->SetDefaultPixelValue( 0 );
	rssResampler->Update();

	m_RSSOutputImage = rssResampler->GetOutput();

	// Get the rigid versor transform from input to atlas
	RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
	RigidTransformType::ParametersType rigidParameters(transform->GetNumberOfParameters());
	rigidParameters[0] = parameters[0];
	rigidParameters[1] = parameters[1];
	rigidParameters[2] = parameters[2];
	rigidParameters[3] = parameters[3];
	rigidParameters[4] = parameters[4];
	rigidParameters[5] = parameters[5];

	rigidTransform->SetParameters(rigidParameters);
	rigidTransform->SetCenter(transform->GetCenter());

	// Resample the rigid registered image
	ResampleFilterType::Pointer rigidResampler = ResampleFilterType::New();
  ResampleInterpolatorType::Pointer rigidResampleInterpolator = ResampleInterpolatorType::New();

	rigidResampler->SetTransform( rigidTransform );
	rigidResampler->SetInput( m_InputImage );
	rigidResampler->SetInterpolator( rigidResampleInterpolator );

	rigidResampler->SetSize(    m_Atlas->GetLargestPossibleRegion().GetSize() );
	rigidResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	rigidResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	rigidResampler->SetOutputDirection( m_Atlas->GetDirection() );
	rigidResampler->SetDefaultPixelValue( 0 );
	rigidResampler->Update();

	m_RigidOutputImage = rigidResampler->GetOutput();

	// Use the Matrix, Offset and Center of rotation of the ScaleSkewVersor3DTransform in order to create an equivalent AffineTransform,
	AffineTransformType::Pointer affineTransform = AffineTransformType::New();
	affineTransform->SetMatrix( transform->GetMatrix() );
	affineTransform->SetCenter( transform->GetCenter() );
	affineTransform->SetOffset( transform->GetOffset() );

	AffineTransformType::Pointer invAffineTransform = AffineTransformType::New();
	affineTransform->GetInverse(invAffineTransform);

	// Rigid x Input = Rigid x inv(A) x Atlas
	DualResampleFilterType::Pointer dualResampler = DualResampleFilterType::New();
	dualResampler->SetInput( m_Atlas );
	dualResampler->SetTransform2( invAffineTransform );
	dualResampler->SetTransform( rigidTransform );
	dualResampler->SetInterpolator( rigidResampleInterpolator );

	dualResampler->SetSize(    m_Atlas->GetLargestPossibleRegion().GetSize() );
	dualResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	dualResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	dualResampler->SetOutputDirection( m_Atlas->GetDirection() );
	dualResampler->SetDefaultPixelValue( 0 );
	dualResampler->Update();

	m_AffineOutputImage = dualResampler->GetOutput();

	// Resample the atlas mask using the resulting affine transform

	DualResampleFilterType::Pointer maskResampler = DualResampleFilterType::New();
	maskResampler->SetTransform2( invAffineTransform );
	maskResampler->SetTransform( rigidTransform );

	maskResampler->SetInput( maskReader->GetOutput() );
	maskResampler->SetInterpolator( rigidResampleInterpolator );

	maskResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	maskResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	maskResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	maskResampler->SetOutputDirection( m_Atlas->GetDirection() );
	maskResampler->SetDefaultPixelValue( 0 );
	maskResampler->Update();

	m_AffineMask = maskResampler->GetOutput();

	// Resample the atlas WM using the resulting affine transform
	typedef itk::LinearInterpolateImageFunction<InternalImageType>	FloatInterpolatorType;
	FloatInterpolatorType::Pointer floatInterpolator = FloatInterpolatorType::New();

	FloatReaderType::Pointer wmReader = FloatReaderType::New();
	std::string wmName = m_AtlasPath + "/mni_icbm152_wm_tal_nlin_sym_09c.nii";
	wmReader->SetFileName(wmName);
	try
	{
		wmReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer wmResampler = FloatResampleFilterType::New();
	wmResampler->SetTransform2( invAffineTransform );
	wmResampler->SetTransform( rigidTransform );

	wmResampler->SetInput( wmReader->GetOutput() );
	wmResampler->SetInterpolator( floatInterpolator );

	wmResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	wmResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	wmResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	wmResampler->SetOutputDirection( m_Atlas->GetDirection() );
	wmResampler->SetDefaultPixelValue( 0 );
	wmResampler->Update();

	m_AffineWM = wmResampler->GetOutput();

	// Resample the atlas WM using the resulting affine transform
	FloatReaderType::Pointer gmReader = FloatReaderType::New();
	std::string gmName = m_AtlasPath + "/mni_icbm152_gm_tal_nlin_sym_09c.nii";
	gmReader->SetFileName(gmName);
	try
	{
		gmReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer gmResampler = FloatResampleFilterType::New();
	gmResampler->SetTransform2( invAffineTransform );
	gmResampler->SetTransform( rigidTransform );

	gmResampler->SetInput( gmReader->GetOutput() );
	gmResampler->SetInterpolator( floatInterpolator );

	gmResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	gmResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	gmResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	gmResampler->SetOutputDirection( m_Atlas->GetDirection() );
	gmResampler->SetDefaultPixelValue( 0 );
	gmResampler->Update();

	m_AffineGM = gmResampler->GetOutput();

	// Resample the atlas CSF using the resulting affine transform
	FloatReaderType::Pointer csfReader = FloatReaderType::New();
	std::string csfName = m_AtlasPath + "/mni_icbm152_csf_tal_nlin_sym_09c.nii";
	csfReader->SetFileName(csfName);
	try
	{
		csfReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	FloatResampleFilterType::Pointer csfResampler = FloatResampleFilterType::New();
	csfResampler->SetTransform2( invAffineTransform );
	csfResampler->SetTransform( rigidTransform );
	csfResampler->SetInput( csfReader->GetOutput() );
	csfResampler->SetInterpolator( floatInterpolator );

	csfResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	csfResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	csfResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	csfResampler->SetOutputDirection( m_Atlas->GetDirection() );
	csfResampler->SetDefaultPixelValue( 0 );
	csfResampler->Update();

	m_AffineCSF = csfResampler->GetOutput();

	// Resample the label using the resulting affine transform
	ReaderType::Pointer labelReader = ReaderType::New();
	std::string labelName = m_AtlasPath + "/mni_label.nii";
	labelReader->SetFileName(labelName);
	try
	{
		labelReader->Update();
	}
	catch(itk::ExceptionObject & err)
	{
		std::cerr<< err <<std::endl;
	}

	typedef itk::NearestNeighborInterpolateImageFunction<ImageType> LabelInterpolatorType;
	LabelInterpolatorType::Pointer labelInterpolator = LabelInterpolatorType::New();

	DualResampleFilterType::Pointer labelResampler = DualResampleFilterType::New();
	labelResampler->SetTransform2( invAffineTransform );
	labelResampler->SetTransform( rigidTransform );

	labelResampler->SetInput( labelReader->GetOutput() );
	labelResampler->SetInterpolator( rigidResampleInterpolator );

	labelResampler->SetSize( m_Atlas->GetLargestPossibleRegion().GetSize() );
	labelResampler->SetOutputOrigin(  m_Atlas->GetOrigin() );
	labelResampler->SetOutputSpacing( m_Atlas->GetSpacing() );
	labelResampler->SetOutputDirection( m_Atlas->GetDirection() );
	labelResampler->SetDefaultPixelValue( 0 );
	labelResampler->Update();

	m_AffineLabel = labelResampler->GetOutput();

}

#endif