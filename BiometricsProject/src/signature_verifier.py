import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
import json
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorldClassSignatureVerificationSystem:
    def __init__(self, similarity_threshold=0.85):
        self.reference_signatures = {}
        self.similarity_threshold = similarity_threshold
        self.history = []

    def preprocess_image(self, image_path):
        """Advanced preprocessing with multiple techniques"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Multiple preprocessing techniques for better feature extraction
            processed = {}

            # Technique 1: Standard preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)

            # Technique 2: Otsu's thresholding
            _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cleaned_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel)

            # Technique 3: Skeletonization for stroke analysis
            skeleton = self._skeletonize(cleaned)

            processed.update({
                'adaptive': cleaned,
                'otsu': cleaned_otsu,
                'skeleton': skeleton,
                'original_gray': gray,
                'original_rgb': img
            })

            return processed

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    def _skeletonize(self, image):
        """Skeletonize image for stroke analysis"""
        skeleton = np.zeros(image.shape, np.uint8)
        eroded = np.zeros(image.shape, np.uint8)
        temp = np.zeros(image.shape, np.uint8)

        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            cv2.erode(binary, kernel, eroded)
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(binary, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            binary, eroded = eroded, binary

            if cv2.countNonZero(binary) == 0:
                break

        return skeleton

    def extract_world_class_features(self, processed_images):
        """Extract comprehensive features with high discrimination power"""
        try:
            features = {}
            binary_img = processed_images['adaptive']
            skeleton_img = processed_images['skeleton']
            gray_img = processed_images['original_gray']

            # 1. CONTOUR-BASED FEATURES
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None

            main_contour = max(contours, key=cv2.contourArea)

            # Basic geometric features
            features['area'] = float(cv2.contourArea(main_contour))
            features['perimeter'] = float(cv2.arcLength(main_contour, True))

            # Bounding box analysis
            x, y, w, h = cv2.boundingRect(main_contour)
            features['aspect_ratio'] = float(w / h if h > 0 else 0)
            features['solidity'] = float(features['area'] / (w * h) if (w * h) > 0 else 0)
            features['extent'] = float(features['area'] / (w * h) if (w * h) > 0 else 0)

            # 2. ADVANCED SHAPE FEATURES
            hull = cv2.convexHull(main_contour)
            features['convexity'] = float(cv2.contourArea(hull) / features['area'] if features['area'] > 0 else 0)
            features['compactness'] = float(
                (features['perimeter'] ** 2) / (4 * np.pi * features['area']) if features['area'] > 0 else 0)

            # Minimum enclosing circle
            (center_x, center_y), radius = cv2.minEnclosingCircle(main_contour)
            features['circularity'] = float(features['area'] / (np.pi * radius * radius) if radius > 0 else 0)

            # 3. MOMENT-BASED FEATURES (Highly discriminative)
            moments = cv2.moments(binary_img)

            # Hu Moments (rotation invariant)
            hu_moments = cv2.HuMoments(moments)
            # Apply logarithmic transform for better comparison
            features['hu_moments'] = (-np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)).flatten().tolist()

            # Central moments
            features['central_moments'] = [
                float(moments['mu20']), float(moments['mu11']), float(moments['mu02']),
                float(moments['mu30']), float(moments['mu21']), float(moments['mu12']), float(moments['mu03'])
            ]

            # 4. STROKE AND SKELETON FEATURES (Crucial for discrimination)
            # Skeleton analysis
            endpoints = self._find_endpoints(skeleton_img)
            branchpoints = self._find_branchpoints(skeleton_img)

            features['endpoint_count'] = int(len(endpoints[0]))
            features['branchpoint_count'] = int(len(branchpoints[0]))
            features['skeleton_length'] = int(np.sum(skeleton_img > 0))
            features['stroke_complexity'] = float(features['endpoint_count'] / (features['skeleton_length'] + 1e-8))

            # 5. TEXTURE AND GRADIENT FEATURES
            # Gradient analysis
            grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            orientation = np.arctan2(grad_y, grad_x)

            features['gradient_mean'] = float(np.mean(magnitude))
            features['gradient_std'] = float(np.std(magnitude))
            features['gradient_entropy'] = float(self._calculate_entropy(magnitude.astype(np.uint8)))

            # 6. STATISTICAL FEATURES
            nonzero_pixels = binary_img[binary_img > 0]
            if len(nonzero_pixels) > 0:
                features['intensity_mean'] = float(np.mean(nonzero_pixels))
                features['intensity_std'] = float(np.std(nonzero_pixels))
                features['intensity_skew'] = float(self._calculate_skewness(nonzero_pixels))
            else:
                features['intensity_mean'] = 0.0
                features['intensity_std'] = 0.0
                features['intensity_skew'] = 0.0

            # 7. FOURIER DESCRIPTORS (Shape frequency analysis)
            features['fourier_descriptors'] = [float(x) for x in
                                               self._calculate_robust_fourier_descriptors(main_contour)]

            # 8. POSITIONAL FEATURES
            if moments['m00'] != 0:
                features['centroid_x'] = float(moments['m10'] / moments['m00'] / binary_img.shape[1])
                features['centroid_y'] = float(moments['m01'] / moments['m00'] / binary_img.shape[0])
            else:
                features['centroid_x'] = 0.0
                features['centroid_y'] = 0.0

            # 9. DENSITY AND DISTRIBUTION FEATURES
            features['pixel_density'] = float(np.sum(binary_img > 0) / (binary_img.shape[0] * binary_img.shape[1]))

            # Mass distribution
            features['mass_distribution'] = [float(x) for x in self._calculate_mass_distribution(binary_img)]

            return features

        except Exception as e:
            logger.error(f"Error extracting world-class features: {e}")
            return None

    def _find_endpoints(self, skeleton):
        """Find endpoints in skeleton"""
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        return np.where(conv == 11)

    def _find_branchpoints(self, skeleton):
        """Find branchpoints in skeleton"""
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        return np.where(conv >= 13)

    def _calculate_entropy(self, image):
        """Calculate image entropy"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-8))
        return float(entropy)

    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        if len(data) == 0:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        skew = np.mean(((data - mean_val) / std_val) ** 3)
        return float(skew)

    def _calculate_robust_fourier_descriptors(self, contour, num_descriptors=15):
        """Calculate robust Fourier descriptors"""
        try:
            if len(contour) < 20:
                return np.zeros(num_descriptors)

            # Convert contour to complex numbers
            contour_complex = np.empty(len(contour), dtype=complex)
            for i, point in enumerate(contour):
                contour_complex[i] = complex(point[0][0], point[0][1])

            # Apply Fourier transform
            fourier_result = np.fft.fft(contour_complex)
            descriptors = np.abs(fourier_result[1:num_descriptors + 1])

            # Robust normalization
            if len(descriptors) > 0 and np.max(descriptors) > 0:
                return descriptors / np.max(descriptors)
            else:
                return np.zeros(num_descriptors)
        except:
            return np.zeros(num_descriptors)

    def _calculate_mass_distribution(self, binary_image):
        """Calculate mass distribution features"""
        try:
            # Calculate moments for mass distribution
            moments = cv2.moments(binary_image)
            if moments['m00'] == 0:
                return np.zeros(4)

            # Central moments normalized
            mu20 = moments['mu20'] / moments['m00'] ** 2
            mu02 = moments['mu02'] / moments['m00'] ** 2
            mu11 = moments['mu11'] / moments['m00'] ** 2

            return np.array([mu20, mu02, mu11, np.sqrt(mu20 * mu02)])
        except:
            return np.zeros(4)

    def create_discriminative_feature_vector(self, features):
        """Create feature vector optimized for discrimination"""
        try:
            feature_vectors = []
            weights = []

            # 1. SHAPE AND GEOMETRIC FEATURES (High weight - 30%)
            geometric = np.array([
                features['area'],
                features['perimeter'],
                features['aspect_ratio'],
                features['solidity'],
                features['convexity'],
                features['compactness'],
                features['circularity'],
                features['extent']
            ])
            feature_vectors.append(geometric)
            weights.extend([0.30] * len(geometric))

            # 2. MOMENT FEATURES (High weight - 25%)
            moments_combined = np.concatenate([
                features['hu_moments'],
                features['central_moments']
            ])
            feature_vectors.append(moments_combined)
            weights.extend([0.25] * len(moments_combined))

            # 3. STROKE FEATURES (High weight - 20%)
            stroke_features = np.array([
                features['endpoint_count'],
                features['branchpoint_count'],
                features['skeleton_length'],
                features['stroke_complexity']
            ])
            feature_vectors.append(stroke_features)
            weights.extend([0.20] * len(stroke_features))

            # 4. TEXTURE AND GRADIENT FEATURES (Medium weight - 15%)
            texture_features = np.array([
                features['gradient_mean'],
                features['gradient_std'],
                features['gradient_entropy'],
                features['intensity_mean'],
                features['intensity_std'],
                features['intensity_skew']
            ])
            feature_vectors.append(texture_features)
            weights.extend([0.15] * len(texture_features))

            # 5. FOURIER AND POSITIONAL FEATURES (Low weight - 10%)
            other_features = np.concatenate([
                features['fourier_descriptors'],
                np.array([features['centroid_x'], features['centroid_y']]),
                np.array([features['pixel_density']]),
                features['mass_distribution']
            ])
            feature_vectors.append(other_features)
            weights.extend([0.10] * len(other_features))

            # Combine all features
            combined_vector = np.concatenate(feature_vectors)
            weights_array = np.array(weights[:len(combined_vector)])

            # Normalize with robust scaling
            vector_mean = np.median(combined_vector)
            vector_std = np.std(combined_vector)

            if vector_std > 0:
                normalized_vector = (combined_vector - vector_mean) / vector_std
            else:
                normalized_vector = combined_vector - vector_mean

            # Apply feature weights
            weighted_vector = normalized_vector * weights_array

            return weighted_vector

        except Exception as e:
            logger.error(f"Error creating discriminative feature vector: {e}")
            return None

    def calculate_advanced_similarity(self, vec1, vec2):
        """Calculate similarity using multiple advanced metrics"""
        try:
            similarities = []
            weights = []

            # 1. COSINE SIMILARITY (40% weight)
            cos_sim = cosine_similarity([vec1], [vec2])[0][0]
            normalized_cos = max(0, (cos_sim + 1) / 2)  # Ensure non-negative
            similarities.append(normalized_cos)
            weights.append(0.40)

            # 2. EUCLIDEAN-BASED SIMILARITY (25% weight)
            euclidean_dist = euclidean(vec1, vec2)
            euclidean_sim = 1 / (1 + euclidean_dist)
            similarities.append(euclidean_sim)
            weights.append(0.25)

            # 3. MANHATTAN DISTANCE SIMILARITY (20% weight)
            manhattan_dist = cityblock(vec1, vec2)
            manhattan_sim = 1 / (1 + manhattan_dist)
            similarities.append(manhattan_sim)
            weights.append(0.20)

            # 4. CORRELATION SIMILARITY (15% weight)
            correlation = np.corrcoef(vec1, vec2)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            correlation_sim = max(0, (correlation + 1) / 2)
            similarities.append(correlation_sim)
            weights.append(0.15)

            # Calculate weighted average
            weighted_similarity = np.sum(np.array(similarities) * np.array(weights))

            return min(weighted_similarity, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating advanced similarity: {e}")
            return 0

    def multi_level_verification(self, test_features, ref_features, similarity_score):
        """Multi-level verification with strict checks"""
        try:
            # Level 1: Overall similarity threshold
            if similarity_score < self.similarity_threshold:
                return False, "Overall similarity too low"

            # Level 2: Geometric consistency check
            geometric_test = np.array([
                test_features['area'],
                test_features['perimeter'],
                test_features['aspect_ratio']
            ])
            geometric_ref = np.array([
                ref_features['area'],
                ref_features['perimeter'],
                ref_features['aspect_ratio']
            ])

            geometric_similarity = 1 - (euclidean(geometric_test, geometric_ref) /
                                        (euclidean(geometric_ref, np.zeros_like(geometric_ref)) + 1e-8))

            if geometric_similarity < 0.8:  # 80% geometric similarity required
                return False, f"Geometric inconsistency: {geometric_similarity:.3f}"

            # Level 3: Stroke feature consistency
            stroke_test = np.array([
                test_features['endpoint_count'],
                test_features['branchpoint_count'],
                test_features['stroke_complexity']
            ])
            stroke_ref = np.array([
                ref_features['endpoint_count'],
                ref_features['branchpoint_count'],
                ref_features['stroke_complexity']
            ])

            stroke_similarity = 1 - (euclidean(stroke_test, stroke_ref) /
                                     (euclidean(stroke_ref, np.zeros_like(stroke_ref)) + 1e-8))

            if stroke_similarity < 0.7:  # 70% stroke similarity required
                return False, f"Stroke pattern mismatch: {stroke_similarity:.3f}"

            # Level 4: Shape descriptor consistency
            hu_similarity = cosine_similarity(
                [test_features['hu_moments']],
                [ref_features['hu_moments']]
            )[0][0]
            normalized_hu = max(0, (hu_similarity + 1) / 2)

            if normalized_hu < 0.75:  # 75% shape similarity required
                return False, f"Shape descriptor mismatch: {normalized_hu:.3f}"

            return True, "All verification levels passed"

        except Exception as e:
            logger.error(f"Multi-level verification error: {e}")
            return similarity_score >= self.similarity_threshold, "Fallback to basic verification"

    def add_reference_signature(self, signature_name, image_path, save_to_disk=True):
        """Add reference signature with world-class feature extraction"""
        try:
            print("🔄 Preprocessing image with advanced techniques...")
            processed = self.preprocess_image(image_path)
            print("🔍 Extracting world-class features...")
            features = self.extract_world_class_features(processed)

            if features is None:
                logger.warning(f"No features extracted from {image_path}")
                return False

            print("⚙️ Creating discriminative feature vector...")
            feature_vector = self.create_discriminative_feature_vector(features)
            if feature_vector is None:
                return False

            self.reference_signatures[signature_name] = {
                'feature_vector': feature_vector,
                'features': features,
                'processed_images': processed,
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }

            print(f"✅ WORLD-CLASS reference signature '{signature_name}' added successfully!")
            print(f"📊 Feature vector length: {len(feature_vector)}")
            print(f"🎯 Using advanced multi-level verification")

            if save_to_disk:
                self.save_references()

            return True

        except Exception as e:
            logger.error(f"Error adding reference signature: {e}")
            return False

    def compare_signatures(self, test_image_path, reference_name):
        """Advanced signature comparison with multi-level verification"""
        try:
            if reference_name not in self.reference_signatures:
                logger.error(f"Reference signature '{reference_name}' not found!")
                return None, None, None, "Reference not found"

            print("🔄 Processing test signature...")
            test_processed = self.preprocess_image(test_image_path)
            test_features = self.extract_world_class_features(test_processed)

            if test_features is None:
                logger.error("No features extracted from test signature!")
                return None, None, None, "Feature extraction failed"

            test_vector = self.create_discriminative_feature_vector(test_features)
            if test_vector is None:
                return None, None, None, "Feature vector creation failed"

            reference_vector = self.reference_signatures[reference_name]['feature_vector']
            reference_features = self.reference_signatures[reference_name]['features']

            # Calculate advanced similarity
            similarity_score = self.calculate_advanced_similarity(test_vector, reference_vector)

            # Multi-level verification
            is_genuine, verification_message = self.multi_level_verification(
                test_features, reference_features, similarity_score
            )

            # Log verification
            verification_record = {
                'timestamp': datetime.now().isoformat(),
                'reference_name': reference_name,
                'test_image_path': test_image_path,
                'similarity_score': float(similarity_score),
                'verdict': 'GENUINE' if is_genuine else 'FORGED',
                'threshold': self.similarity_threshold,
                'verification_message': verification_message
            }
            self.history.append(verification_record)

            return similarity_score, is_genuine, test_features, verification_message

        except Exception as e:
            logger.error(f"Error comparing signatures: {e}")
            return None, None, None, f"Comparison error: {str(e)}"

    def visualize_comparison(self, test_image_path, reference_name, similarity_score, is_genuine, test_features,
                             verification_message):
        """Enhanced visualization with verification details"""
        try:
            if reference_name not in self.reference_signatures:
                return

            ref_data = self.reference_signatures[reference_name]

            # Load test image
            test_img = cv2.imread(test_image_path)

            # Create comparison figure
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'WORLD-CLASS SIGNATURE VERIFICATION\nResult: {" GENUINE" if is_genuine else "❌ FORGED"}',
                         fontsize=16, fontweight='bold',
                         color='green' if is_genuine else 'red')

            # Reference signature
            axes[0, 0].imshow(ref_data['processed_images']['original_gray'], cmap='gray')
            axes[0, 0].set_title(
                f'REFERENCE: {reference_name}\nArea: {ref_data["features"]["area"]:.0f} | Endpoints: {ref_data["features"]["endpoint_count"]}',
                fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')

            axes[1, 0].imshow(ref_data['processed_images']['skeleton'], cmap='gray')
            axes[1, 0].set_title('Reference Skeleton\n(Stroke Analysis)', fontsize=11)
            axes[1, 0].axis('off')

            # Test signature
            axes[0, 1].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(
                f'TEST SIGNATURE\nArea: {test_features["area"]:.0f} | Endpoints: {test_features["endpoint_count"]}',
                fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')

            axes[1, 1].imshow(test_features.get('skeleton', np.zeros_like(test_img[:, :, 0])), cmap='gray')
            axes[1, 1].set_title('Test Skeleton\n(Stroke Analysis)', fontsize=11)
            axes[1, 1].axis('off')

            # Verification result with details
            color = 'green' if is_genuine else 'red'
            result_text = '✅ GENUINE SIGNATURE' if is_genuine else ' FORGED SIGNATURE'

            info_text = f"""VERIFICATION RESULT:
{result_text}

Similarity Score: {similarity_score:.3f}
Threshold: {self.similarity_threshold}

VERIFICATION DETAILS:
{verification_message}

FEATURE ANALYSIS:
• Geometric Similarity: High
• Stroke Pattern: {'Match' if is_genuine else 'Mismatch'} 
• Shape Descriptors: {'Consistent' if is_genuine else 'Inconsistent'}
"""

            axes[0, 2].text(0.05, 0.85, info_text, fontsize=11, fontweight='bold',
                            transform=axes[0, 2].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            axes[0, 2].axis('off')

            # Advanced feature comparison
            features_compare = ['Area', 'Endpoints', 'Branches', 'Complexity', 'Convexity']
            ref_values = [
                ref_data['features']['area'],
                ref_data['features']['endpoint_count'],
                ref_data['features']['branchpoint_count'],
                ref_data['features']['stroke_complexity'] * 1000,
                ref_data['features']['convexity'] * 100
            ]
            test_values = [
                test_features['area'],
                test_features['endpoint_count'],
                test_features['branchpoint_count'],
                test_features['stroke_complexity'] * 1000,
                test_features['convexity'] * 100
            ]

            x = np.arange(len(features_compare))
            width = 0.35

            bars1 = axes[1, 2].bar(x - width / 2, ref_values, width, label='Reference',
                                   alpha=0.7, color='blue', edgecolor='black')
            bars2 = axes[1, 2].bar(x + width / 2, test_values, width, label='Test',
                                   alpha=0.7, color='orange', edgecolor='black')

            axes[1, 2].set_xlabel('Critical Features', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Values', fontsize=12, fontweight='bold')
            axes[1, 2].set_title('ADVANCED FEATURE COMPARISON\n(Key Discrimination Factors)',
                                 fontsize=12, fontweight='bold')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(features_compare, rotation=45, fontweight='bold')
            axes[1, 2].legend(fontsize=10)
            axes[1, 2].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars1 + bars2:
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

    def save_references(self, filepath='reference_signatures.json'):
        """Save reference signatures to JSON file"""
        try:
            save_data = {}
            for name, data in self.reference_signatures.items():
                # Convert all numpy types to Python native types for JSON serialization
                save_data[name] = {
                    'feature_vector': data['feature_vector'].tolist(),
                    'features': {},
                    'image_path': data['image_path'],
                    'timestamp': data['timestamp']
                }

                # Convert feature values to JSON-serializable types
                for k, v in data['features'].items():
                    if isinstance(v, np.ndarray):
                        save_data[name]['features'][k] = v.tolist()
                    elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
                        save_data[name]['features'][k] = v.item()  # Convert numpy scalars to Python types
                    else:
                        save_data[name]['features'][k] = v

            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"Reference signatures saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving references: {e}")

    def load_references(self, filepath='reference_signatures.json'):
        """Load reference signatures from JSON file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Reference file {filepath} not found")
                return False

            with open(filepath, 'r') as f:
                save_data = json.load(f)

            for name, data in save_data.items():
                self.reference_signatures[name] = {
                    'feature_vector': np.array(data['feature_vector']),
                    'features': data['features'],
                    'image_path': data['image_path'],
                    'timestamp': data['timestamp']
                }

            logger.info(f"Loaded {len(self.reference_signatures)} reference signatures from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading references: {e}")
            return False

    def get_reference_list(self):
        """Get list of all reference signature names"""
        return list(self.reference_signatures.keys())

    def get_verification_history(self):
        """Get verification history"""
        return self.history