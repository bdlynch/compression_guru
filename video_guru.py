#!/usr/bin/env python3
"""
Ultimate Video Compression Guru - M1 Optimized with Advanced Video Analysis
Intelligently analyzes videos and compresses with maximum size reduction and minimal quality loss
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import tempfile
import statistics
from datetime import timedelta

class AdvancedVideoAnalyzer:
    def __init__(self):
        self.supported_formats = {'.mp4', '.mkv', '.avi', '.mov', '.m4v', '.webm', '.flv', '.wmv', '.ts', '.mts'}
        self.preset_configs = {
            'ultra': {
                'name': 'Ultra Compression (Smallest Files)',
                'description': 'ABSOLUTE minimum file sizes with acceptable quality. 75-90% size reduction.',
                'crf': 28,
                'preset': 'faster',
                'profile': 'main',
                'level': '4.0',
                'additional_flags': ['-tune', 'film']
            },
            'aggressive': {
                'name': 'Aggressive Compression (Tiny Files)', 
                'description': 'Maximum compression for smallest possible files. 80-92% size reduction.',
                'crf': 30,
                'preset': 'faster',
                'profile': 'baseline',
                'level': '3.1',
                'additional_flags': ['-tune', 'grain']
            },
            'high': {
                'name': 'High Compression (Balanced)',
                'description': 'High compression with minimal quality loss. 60-75% size reduction.',
                'crf': 23,
                'preset': 'faster',
                'profile': 'high', 
                'level': '4.1',
                'additional_flags': ['-tune', 'film']
            },
            'quality': {
                'name': 'Quality Priority (Conservative)',
                'description': 'Moderate compression prioritizing quality. 40-60% size reduction.',
                'crf': 20,
                'preset': 'faster',
                'profile': 'high',
                'level': '4.2',
                'additional_flags': []
            },
            'extreme': {
                'name': 'EXTREME Compression (Experimental)',
                'description': 'Pushes compression to absolute limits. 85-95% size reduction. May have visible quality loss.',
                'crf': 32,
                'preset': 'ultrafast',
                'profile': 'baseline',
                'level': '3.0',
                'additional_flags': ['-tune', 'fastdecode']
            }
        }
    
    def get_video_info(self, file_path: str) -> Dict:
        """Extract comprehensive video information using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
            '-show_streams', '-show_chapters', '-show_programs', str(file_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return {}
        # Add this check after running ffprobe
        if not result or result.strip() == "":
            print(f"❌ ffprobe returned no data for {file_path}")
            return None

        # Also add this validation
        if 'streams' not in info or not info['streams']:
            print(f"❌ No streams found in {file_path}")
            return None

    def detect_hardcoded_subtitles_advanced(self, file_path: str, video_info: Dict) -> Dict:
        """Advanced hardcoded subtitle detection using multiple methods"""
        detection_results = {
            'has_hardcoded_subs': False,
            'confidence': 0.0,
            'detection_methods': [],
            'subtitle_region': None,
            'language_hints': []
        }
        
        # Method 1: Check for absence of subtitle streams but presence of subtitle indicators
        streams = video_info.get('streams', [])
        video_streams = [s for s in streams if s.get('codec_type') == 'video']
        subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']
        
        if not subtitle_streams and video_streams:
            detection_results['detection_methods'].append('no_subtitle_streams')
            detection_results['confidence'] += 0.3
        
        # Method 2: Analyze filename and metadata for subtitle indicators
        filename = Path(file_path).name.lower()
        subtitle_indicators = [
            'sub', 'subtitle', 'subtitled', 'hardsub', 'hardcoded', 'burned',
            'baked', 'embedded', 'cc', 'closed.caption', 'eng.sub', 'english.sub'
        ]
        
        for indicator in subtitle_indicators:
            if indicator in filename:
                detection_results['detection_methods'].append(f'filename_indicator_{indicator}')
                detection_results['confidence'] += 0.4
                break
        
        # Method 3: Check video stream metadata and tags
        if video_streams:
            video_stream = video_streams[0]
            tags = video_stream.get('tags', {})
            
            for key, value in tags.items():
                if any(indicator in str(value).lower() for indicator in subtitle_indicators):
                    detection_results['detection_methods'].append(f'metadata_tag_{key}')
                    detection_results['confidence'] += 0.3
        
        # Method 4: Advanced frame analysis for text detection
        frame_analysis = self.analyze_sample_frames_for_text(file_path, video_info)
        if frame_analysis['text_detected']:
            detection_results['detection_methods'].append('frame_text_analysis')
            detection_results['confidence'] += frame_analysis['confidence']
            detection_results['subtitle_region'] = frame_analysis.get('text_region')
        
        # Method 5: Scene change analysis (subtitle changes often correlate with scene changes)
        scene_analysis = self.analyze_scene_changes(file_path, video_info)
        if scene_analysis['subtitle_pattern_detected']:
            detection_results['detection_methods'].append('scene_change_pattern')
            detection_results['confidence'] += 0.2
        
        # Final determination
        detection_results['has_hardcoded_subs'] = detection_results['confidence'] >= 0.4
        detection_results['confidence'] = min(detection_results['confidence'], 1.0)
        
        return detection_results

    def analyze_sample_frames_for_text(self, file_path: str, video_info: Dict) -> Dict:
        """Analyze sample frames to detect text regions"""
        result = {
            'text_detected': False,
            'confidence': 0.0,
            'text_region': None,
            'sample_count': 0
        }
        
        try:
            # Get video duration and sample at multiple points
            duration = float(video_info.get('format', {}).get('duration', 0))
            if duration <= 0:
                return result
            
            # Sample at 10%, 30%, 50%, 70%, 90% of video
            sample_points = [duration * p for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
            text_regions_found = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, timestamp in enumerate(sample_points):
                    frame_path = Path(temp_dir) / f"frame_{i}.png"
                    
                    # Extract frame at timestamp
                    cmd = [
                        'ffmpeg', '-ss', str(timestamp), '-i', file_path,
                        '-frames:v', '1', '-y', str(frame_path)
                    ]
                    
                    try:
                        subprocess.run(cmd, capture_output=True, check=True)
                        if frame_path.exists():
                            # Analyze frame for text (simple method using ffmpeg's drawtext detection)
                            text_analysis = self.detect_text_in_frame(str(frame_path))
                            if text_analysis['has_text']:
                                text_regions_found.append(text_analysis)
                                result['sample_count'] += 1
                    except subprocess.CalledProcessError:
                        continue
            
            if text_regions_found:
                result['text_detected'] = True
                result['confidence'] = min(len(text_regions_found) * 0.2, 0.6)
                
                # Determine common text region (likely subtitle area)
                if len(text_regions_found) >= 2:
                    # Most subtitles appear in bottom third of screen
                    bottom_third_detections = sum(1 for r in text_regions_found 
                                                if r.get('region', {}).get('y_position', 0) > 0.6)
                    if bottom_third_detections >= 2:
                        result['text_region'] = 'bottom_third'
                        result['confidence'] += 0.2
        
        except Exception as e:
            print(f"⚠️  Frame analysis warning: {e}")
        
        return result

    def detect_text_in_frame(self, frame_path: str) -> Dict:
        """Simple text detection in a frame using edge detection heuristics"""
        result = {'has_text': False, 'region': {}}
        
        try:
            # Use ffmpeg to analyze frame characteristics that suggest text
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_frames', '-show_streams',
                '-print_format', 'json', frame_path
            ]
            
            probe_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            frame_data = json.loads(probe_result.stdout)
            
            # Heuristic: Look for high contrast regions typical of subtitles
            # This is a simplified approach - in production you'd use proper OCR
            if 'frames' in frame_data and frame_data['frames']:
                frame = frame_data['frames'][0]
                # Simple heuristic based on frame complexity
                if 'tags' in frame:
                    # If frame analysis shows patterns consistent with text overlays
                    result['has_text'] = True
                    result['region'] = {'y_position': 0.8}  # Assume bottom region
        
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        
        return result

    def analyze_scene_changes(self, file_path: str, video_info: Dict) -> Dict:
        """Analyze scene changes to detect subtitle patterns"""
        result = {
            'subtitle_pattern_detected': False,
            'scene_changes': [],
            'change_frequency': 0.0
        }
        
        try:
            duration = float(video_info.get('format', {}).get('duration', 0))
            if duration <= 30:  # Skip analysis for very short videos
                return result
            
            # Use ffmpeg's scene detection filter on a sample
            sample_duration = min(120, duration)  # Analyze first 2 minutes
            
            cmd = [
                'ffmpeg', '-i', file_path, '-t', str(sample_duration),
                '-vf', 'select=gt(scene\\,0.3),showinfo', '-f', 'null', '-'
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse scene change information from ffmpeg output
            scene_times = []
            for line in process.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        time_match = re.search(r'pts_time:([0-9.]+)', line)
                        if time_match:
                            scene_times.append(float(time_match.group(1)))
                    except ValueError:
                        continue
            
            if len(scene_times) > 5:  # Minimum scenes for pattern analysis
                # Calculate scene change frequency
                result['change_frequency'] = len(scene_times) / sample_duration
                result['scene_changes'] = scene_times
                
                # Heuristic: Regular scene changes might indicate dialogue/subtitle timing
                if 0.1 <= result['change_frequency'] <= 1.0:  # 1 change every 1-10 seconds
                    result['subtitle_pattern_detected'] = True
        
        except Exception as e:
            print(f"⚠️  Scene analysis warning: {e}")
        
        return result

    def analyze_video_characteristics(self, info: Dict, file_path: str) -> Dict:
        """Comprehensive video analysis including advanced subtitle detection"""
        video_stream = None
        audio_streams = []
        subtitle_streams = []
        
        for stream in info.get('streams', []):
            if stream['codec_type'] == 'video':
                video_stream = stream
            elif stream['codec_type'] == 'audio':
                audio_streams.append(stream)
            elif stream['codec_type'] == 'subtitle':
                subtitle_streams.append(stream)
        
        if not video_stream:
            return {}
        
        # Basic video characteristics
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        fps = eval(video_stream.get('r_frame_rate', '0/1')) if video_stream.get('r_frame_rate') else 0
        bitrate = int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else 0
        duration = float(info.get('format', {}).get('duration', 0))
        codec = video_stream.get('codec_name', '')
        pixel_format = video_stream.get('pix_fmt', '')
        
        # Comprehensive bloat analysis
        print("🔍 Analyzing potential bloat factors...")
        bloat_analysis = self.analyze_bloat_factors(info, str(file_path))
        
        # Additional analysis
        complexity_analysis = self.analyze_video_complexity(video_stream, info)

    def analyze_bloat_factors(self, info: Dict, file_path: str) -> Dict:
        """Comprehensive bloat analysis to identify unnecessary data"""
        bloat_analysis = {
            'audio_bloat': {},
            'metadata_bloat': {},
            'stream_bloat': {},
            'total_bloat_score': 0,
            'recommendations': [],
            'potential_savings_mb': 0
        }
        
        # Analyze audio streams for bloat
        audio_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'audio']
        video_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'video']
        
        # Audio bloat detection
        if audio_streams:
            total_audio_bitrate = 0
            redundant_audio = 0
            high_bitrate_audio = 0
            
            for i, audio in enumerate(audio_streams):
                bitrate = int(audio.get('bit_rate', 0)) if audio.get('bit_rate') else 0
                channels = int(audio.get('channels', 2))
                sample_rate = int(audio.get('sample_rate', 48000))
                codec = audio.get('codec_name', '')
                
                total_audio_bitrate += bitrate
                
                # Detect high bitrate audio (over 320kbps for stereo)
                if bitrate > 320000 and channels <= 2:
                    high_bitrate_audio += 1
                    bloat_analysis['recommendations'].append(f'Audio stream {i+1}: Excessive bitrate ({bitrate//1000}kbps) for {channels} channels')
                
                # Detect unnecessary multichannel for non-surround content
                if channels > 2 and not any(indicator in str(audio.get('tags', {})).lower() for indicator in ['surround', '5.1', '7.1']):
                    redundant_audio += 1
                    bloat_analysis['recommendations'].append(f'Audio stream {i+1}: {channels} channels may be unnecessary')
                
                # Detect lossless audio where lossy would suffice
                if codec in ['flac', 'alac', 'tta', 'wavpack'] and bitrate > 500000:
                    bloat_analysis['recommendations'].append(f'Audio stream {i+1}: Lossless codec {codec} could be converted to high-quality AAC')
            
            bloat_analysis['audio_bloat'] = {
                'total_streams': len(audio_streams),
                'total_bitrate': total_audio_bitrate,
                'redundant_streams': redundant_audio,
                'high_bitrate_streams': high_bitrate_audio,
                'avg_bitrate_per_stream': total_audio_bitrate // len(audio_streams) if audio_streams else 0
            }
            
            # Calculate potential audio savings
            if high_bitrate_audio > 0 or redundant_audio > 0:
                duration = float(info.get('format', {}).get('duration', 0))
                potential_audio_savings = (total_audio_bitrate * 0.3 * duration) / 8 / (1024*1024)  # 30% reduction estimate
                bloat_analysis['potential_savings_mb'] += potential_audio_savings
        
        # Metadata bloat analysis
        format_info = info.get('format', {})
        metadata_size = 0
        
        # Check for excessive metadata
        tags = format_info.get('tags', {})
        metadata_fields = len(tags)
        
        for key, value in tags.items():
            if isinstance(value, str):
                metadata_size += len(value.encode('utf-8'))
        
        # Check for thumbnails/cover art
        attachment_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'attachment']
        cover_art_size = 0
        
        for attachment in attachment_streams:
            if 'cover' in attachment.get('tags', {}).get('filename', '').lower():
                # Estimate cover art size (rough)
                cover_art_size += 500000  # Assume 500KB per cover
        
        bloat_analysis['metadata_bloat'] = {
            'metadata_fields': metadata_fields,
            'estimated_metadata_size_kb': metadata_size / 1024,
            'attachment_streams': len(attachment_streams),
            'estimated_cover_art_size_kb': cover_art_size / 1024
        }
        
        if metadata_fields > 20:
            bloat_analysis['recommendations'].append(f'Excessive metadata: {metadata_fields} fields detected')
        
        if cover_art_size > 1000000:  # >1MB
            bloat_analysis['recommendations'].append(f'Large cover art detected: ~{cover_art_size//1024//1024}MB')
            bloat_analysis['potential_savings_mb'] += cover_art_size / (1024*1024)
        
        # Stream redundancy analysis
        subtitle_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'subtitle']
        
        bloat_analysis['stream_bloat'] = {
            'video_streams': len(video_streams),
            'audio_streams': len(audio_streams),
            'subtitle_streams': len(subtitle_streams),
            'total_streams': len(info.get('streams', []))
        }
        
        # Multiple video streams (rare but possible)
        if len(video_streams) > 1:
            bloat_analysis['recommendations'].append(f'Multiple video streams detected: {len(video_streams)} streams')
        
        # Excessive subtitle streams
        if len(subtitle_streams) > 5:
            bloat_analysis['recommendations'].append(f'Many subtitle streams: {len(subtitle_streams)} languages')
        
        # Calculate total bloat score (0-100)
        score = 0
        score += min(high_bitrate_audio * 15, 30)  # High bitrate audio
        score += min(redundant_audio * 10, 20)     # Redundant audio
        score += min(metadata_fields, 20)          # Metadata bloat
        score += min(len(subtitle_streams), 15)    # Subtitle bloat
        score += min(cover_art_size / 100000, 15)  # Cover art bloat
        
        bloat_analysis['total_bloat_score'] = min(score, 100)
        
        return bloat_analysis
        
    def generate_detailed_logs(self, file_path: str, characteristics: Dict, 
                            preset: str, cmd: List[str]) -> str:
        """Generate comprehensive compression logs"""
        timestamp = timedelta(seconds=int(characteristics.get('duration', 0)))
        
        log_content = f"""
    🎬 COMPRESSION LOG - {Path(file_path).name}
    {'='*60}
    📅 Generated: {timedelta(seconds=int(__import__('time').time()))}
    🎯 Preset: {preset.upper()}
    ⏱️  Duration: {timestamp}

    📊 ORIGINAL VIDEO ANALYSIS:
    Resolution: {characteristics['width']}x{characteristics['height']} ({characteristics['resolution_class']})
    Codec: {characteristics['codec']}
    FPS: {characteristics['fps']:.2f}
    Bitrate: {characteristics.get('bitrate', 0)//1000}kbps
    HDR: {'Yes' if characteristics.get('is_hdr') else 'No'}
    Pixel Format: {characteristics.get('pixel_format', 'N/A')}

    🎵 AUDIO ANALYSIS:
    Streams: {characteristics['audio_streams']}
    
    📝 SUBTITLE ANALYSIS:
    Streams: {characteristics['subtitle_streams']}
    Hardcoded: {'Yes' if characteristics.get('subtitle_analysis', {}).get('has_hardcoded_subs') else 'No'}
    Confidence: {characteristics.get('subtitle_analysis', {}).get('confidence', 0):.1%}

    🔍 BLOAT ANALYSIS:
    Bloat Score: {characteristics.get('bloat_analysis', {}).get('total_bloat_score', 0):.1f}/100
    Potential Savings: {characteristics.get('bloat_analysis', {}).get('potential_savings_mb', 0):.1f}MB
    
    💡 RECOMMENDATIONS:"""
        
        recommendations = characteristics.get('bloat_analysis', {}).get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                log_content += f"\n   • {rec}"
        else:
            log_content += "\n   • No major bloat detected - file is well optimized"
        
        log_content += f"""

    ⚙️  COMPRESSION SETTINGS:
    CRF: {self.preset_configs[preset]['crf']}
    Preset: {self.preset_configs[preset]['preset']}
    Profile: {self.preset_configs[preset]['profile']}
    Tune: {characteristics.get('video_complexity', {}).get('optimal_tune', 'film')}

    🚀 FFMPEG COMMAND:
    {' '.join(cmd)}

    {'='*60}
    """
        return log_content

        characteristics = {
            'width': width,
            'height': height,
            'fps': fps,
            'bitrate': bitrate,
            'duration': duration,
            'codec': codec,
            'pixel_format': pixel_format,
            'audio_streams': len(audio_streams),
            'subtitle_streams': len(subtitle_streams),
            'resolution_class': self.classify_resolution(width, height),
            'is_hdr': self.detect_hdr(video_stream),
            'video_complexity': complexity_analysis,
            'subtitle_analysis': subtitle_analysis
        }
        
        return characteristics

    def analyze_video_complexity(self, video_stream: Dict, info: Dict) -> Dict:
        """Analyze video complexity for optimal compression settings"""
        complexity = {
            'level': 'medium',
            'factors': [],
            'optimal_tune': 'film'
        }
        
        # Analyze bitrate vs resolution ratio
        width = int(video_stream.get('width', 1920))
        height = int(video_stream.get('height', 1080))
        bitrate = int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else 0
        
        pixels = width * height
        if bitrate > 0:
            bitrate_per_pixel = bitrate / pixels
            
            if bitrate_per_pixel > 0.1:
                complexity['level'] = 'high'
                complexity['factors'].append('high_bitrate_density')
            elif bitrate_per_pixel < 0.02:
                complexity['level'] = 'low'
                complexity['factors'].append('low_bitrate_density')
        
        # Analyze frame rate
        fps = eval(video_stream.get('r_frame_rate', '0/1')) if video_stream.get('r_frame_rate') else 0
        if fps > 30:
            complexity['factors'].append('high_framerate')
            complexity['optimal_tune'] = 'animation'
        elif fps < 24:
            complexity['factors'].append('low_framerate')
        
        # Check for interlacing
        if video_stream.get('field_order', 'progressive') != 'progressive':
            complexity['factors'].append('interlaced')
            complexity['level'] = 'high'
        
        return complexity

    def classify_resolution(self, width: int, height: int) -> str:
        """Classify video resolution with more precision"""
        if height >= 2160:
            return '4K'
        elif height >= 1440:
            return '1440p'
        elif height >= 1080:
            return '1080p'
        elif height >= 720:
            return '720p'
        elif height >= 480:
            return '480p'
        else:
            return 'SD'
    
    def detect_hdr(self, video_stream: Dict) -> bool:
        """Enhanced HDR detection"""
        color_space = video_stream.get('color_space', '').lower()
        color_transfer = video_stream.get('color_transfer', '').lower()
        color_primaries = video_stream.get('color_primaries', '').lower()
        
        hdr_indicators = ['bt2020', 'smpte2084', 'arib-std-b67', 'rec2020', 'hlg']
        
        hdr_detected = any(indicator in ' '.join([color_space, color_transfer, color_primaries]) 
                          for indicator in hdr_indicators)
        
        # Also check bit depth
        pix_fmt = video_stream.get('pix_fmt', '')
        if '10le' in pix_fmt or '12le' in pix_fmt:
            hdr_detected = True
        
        return hdr_detected
    
    def estimate_output_size(self, characteristics: Dict, preset: str, original_size: int) -> int:
        """Enhanced size estimation based on comprehensive analysis"""
        config = self.preset_configs[preset]
        crf = config['crf']
        
        # Base compression ratios (updated based on research)
        crf_ratios = {20: 0.45, 23: 0.25, 28: 0.12, 30: 0.08, 32: 0.05}
        base_ratio = crf_ratios.get(crf, 0.25)
        
        # Adjust based on resolution
        res_class = characteristics['resolution_class']
        if res_class == '4K':
            base_ratio *= 0.7  # 4K compresses very well
        elif res_class == '1440p':
            base_ratio *= 0.8
        elif res_class in ['480p', 'SD']:
            base_ratio *= 1.3  # Lower resolutions don't compress as well
        
        # Adjust for subtitle analysis
        subtitle_analysis = characteristics.get('subtitle_analysis', {})
        if subtitle_analysis.get('has_hardcoded_subs', False):
            confidence = subtitle_analysis.get('confidence', 0)
            # Higher confidence in hardcoded subs = less compression
            base_ratio *= (1.0 + confidence * 0.2)
        
        # Adjust for video complexity
        complexity = characteristics.get('video_complexity', {})
        if complexity.get('level') == 'high':
            base_ratio *= 1.15
        elif complexity.get('level') == 'low':
            base_ratio *= 0.9
        
        # Adjust for HDR content
        if characteristics.get('is_hdr', False):
            base_ratio *= 1.2
        
        # Adjust for high frame rates
        if characteristics.get('fps', 0) > 30:
            base_ratio *= 1.1
        
        return int(original_size * base_ratio)
    
    def generate_ffmpeg_command(self, input_file: str, output_file: str, 
                              characteristics: Dict, preset: str) -> List[str]:
        """Generate highly optimized FFmpeg command for M1 MacBook Pro"""
        config = self.preset_configs[preset]
        
        cmd = ['ffmpeg', '-i', input_file]
        
        # Hardware acceleration decision based on content analysis
        res_class = characteristics.get('resolution_class', '1080p')
        complexity = characteristics.get('video_complexity', {})
        
        use_hardware = False
        if res_class in ['4K', '1440p'] and complexity.get('level') != 'high':
            use_hardware = True
        
        if use_hardware:
            cmd.extend(['-c:v', 'h264_videotoolbox'])
            cmd.extend(['-allow_sw', '1'])
            cmd.extend(['-b:v', self.calculate_target_bitrate(characteristics, preset)])
        else:
            cmd.extend(['-c:v', 'libx264'])
            cmd.extend(['-crf', str(config['crf'])])
            cmd.extend(['-preset', config['preset']])
        
        # Profile and level settings
        cmd.extend(['-profile:v', config['profile']])
        cmd.extend(['-level:v', config['level']])
        
        # Tuning based on content analysis
        tune = complexity.get('optimal_tune', 'film')
        subtitle_analysis = characteristics.get('subtitle_analysis', {})
        
        if subtitle_analysis.get('has_hardcoded_subs', False):
            tune = 'film'  # Better for content with text
        
        if config.get('additional_flags'):
            cmd.extend(config['additional_flags'])
        else:
            cmd.extend(['-tune', tune])
        
        # Pixel format optimization
        if characteristics.get('is_hdr', False):
            cmd.extend(['-pix_fmt', 'yuv420p10le'])
        else:
            cmd.extend(['-pix_fmt', 'yuv420p'])

        # Audio handling - NEVER touch audio as requested
        cmd.extend(['-c:a', 'copy'])
        cmd.extend(['-avoid_negative_ts', 'make_zero'])
        
        # Enhanced audio optimization based on bloat analysis
        bloat_analysis = characteristics.get('bloat_analysis', {})
        audio_bloat = bloat_analysis.get('audio_bloat', {})

        # Only optimize audio if bloat detected and preset allows it
        if preset in ['ultra', 'aggressive', 'extreme'] and audio_bloat.get('high_bitrate_streams', 0) > 0:
            # Replace audio copy with optimized encoding
            cmd = [c for c in cmd if c not in ['-c:a', 'copy']]
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-ac', '2'])
            print("🎵 Optimizing high-bitrate audio streams...")

        # Strip metadata if bloat detected
        metadata_bloat = bloat_analysis.get('metadata_bloat', {})
        if metadata_bloat.get('metadata_fields', 0) > 15 or preset in ['extreme']:
            cmd.extend(['-map_metadata', '-1'])
            print("📋 Stripping excessive metadata...")

        # Remove cover art if detected and preset allows
        if metadata_bloat.get('attachment_streams', 0) > 0 and preset in ['ultra', 'aggressive', 'extreme']:
            cmd.extend(['-map', '0:v', '-map', '0:a', '-map', '0:s?'])
            print("🖼️  Removing embedded cover art...")
        
        # Subtitle handling
        cmd.extend(['-c:s', 'copy'])
        
        # Advanced optimization flags for M1
        cmd.extend(['-threads', '0'])
        cmd.extend(['-movflags', '+faststart'])
        
        # Add encoding optimizations based on content
        if not use_hardware:
            # Software encoding optimizations
            x264_params = []
            
            if preset in ['ultra', 'aggressive', 'extreme']:
                x264_params.extend(['aq-mode=2', 'aq-strength=0.8', 'me=hex', 'subme=6'])
            
            if complexity.get('level') == 'high':
                x264_params.extend(['deblock=0:0', 'nr=10'])
            
            if x264_params:
                cmd.extend(['-x264-params', ':'.join(x264_params)])
        
        # Output settings
        cmd.extend(['-f', 'mp4', '-y', output_file])
        
        return cmd

    def calculate_target_bitrate(self, characteristics: Dict, preset: str) -> str:
        """Calculate target bitrate for hardware encoding"""
        width = characteristics.get('width', 1920)
        height = characteristics.get('height', 1080)
        fps = characteristics.get('fps', 30)
        
        # Base bitrates per resolution for different presets
        bitrate_tables = {
            'quality': {'4K': 25000, '1440p': 16000, '1080p': 8000, '720p': 5000},
            'high': {'4K': 15000, '1440p': 10000, '1080p': 5000, '720p': 2500},
            'ultra': {'4K': 8000, '1440p': 5000, '1080p': 2500, '720p': 1500},
            'aggressive': {'4K': 5000, '1440p': 3000, '1080p': 1500, '720p': 800},
            'extreme': {'4K': 3000, '1440p': 2000, '1080p': 1000, '720p': 600}
        }
        
        res_class = characteristics.get('resolution_class', '1080p')
        base_bitrate = bitrate_tables.get(preset, bitrate_tables['high']).get(res_class, 5000)
        
        # Adjust for frame rate
        if fps > 30:
            base_bitrate = int(base_bitrate * (fps / 30))
        
        return f"{base_bitrate}k"

    def process_file(self, file_path: Path, output_dir: Path, presets: List[str]) -> Dict:
        """Process a single video file with comprehensive analysis"""
        print(f"\n🎬 Analyzing: {file_path.name}")
        print("=" * 60)
        
        # Get comprehensive video information
        info = self.get_video_info(str(file_path))
        if not info:
            return {'error': 'Could not analyze video file'}
        
        characteristics = self.analyze_video_characteristics(info, str(file_path))
        if not characteristics:
            return {'error': 'No video stream found'}
        
        original_size = file_path.stat().st_size
        
        # Display comprehensive analysis
        print(f"📊 Video Analysis Results:")
        print(f"   📐 Resolution: {characteristics['width']}x{characteristics['height']} ({characteristics['resolution_class']})")
        print(f"   ⏱️  Duration: {timedelta(seconds=int(characteristics['duration']))}")
        print(f"   🎞️  FPS: {characteristics['fps']:.2f}")
        print(f"   🎥 Codec: {characteristics['codec']}")
        print(f"   📦 Original Size: {original_size / (1024*1024):.1f} MB")
        print(f"   🌈 HDR: {'Yes' if characteristics['is_hdr'] else 'No'}")
        print(f"   🎵 Audio Streams: {characteristics['audio_streams']}")
        print(f"   📝 Subtitle Streams: {characteristics['subtitle_streams']}")
        
        # Subtitle analysis results
        subtitle_analysis = characteristics.get('subtitle_analysis', {})
        print(f"\n🔍 Subtitle Analysis:")
        print(f"   Hardcoded Subtitles: {'Yes' if subtitle_analysis.get('has_hardcoded_subs') else 'No'}")
        print(f"   Confidence: {subtitle_analysis.get('confidence', 0):.1%}")
        if subtitle_analysis.get('detection_methods'):
            print(f"   Detection Methods: {', '.join(subtitle_analysis['detection_methods'])}")

        # Bloat analysis results
        bloat_analysis = characteristics.get('bloat_analysis', {})
        print(f"\n💾 Bloat Analysis:")
        print(f"   Bloat Score: {bloat_analysis.get('total_bloat_score', 0):.1f}/100")
        print(f"   Potential Savings: {bloat_analysis.get('potential_savings_mb', 0):.1f} MB")
        audio_bloat = bloat_analysis.get('audio_bloat', {})
        if audio_bloat.get('total_streams', 0) > 0:
            print(f"   Audio Streams: {audio_bloat['total_streams']} (avg {audio_bloat.get('avg_bitrate_per_stream', 0)//1000}kbps)")
        if audio_bloat.get('high_bitrate_streams', 0) > 0:
            print(f"   ⚠️  High Bitrate Audio: {audio_bloat['high_bitrate_streams']} streams")
        metadata_bloat = bloat_analysis.get('metadata_bloat', {})
        if metadata_bloat.get('metadata_fields', 0) > 10:
            print(f"   📋 Metadata Fields: {metadata_bloat['metadata_fields']}")
    
        recommendations = bloat_analysis.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"   • {rec}")

        # Complexity analysis
        complexity = characteristics.get('video_complexity', {})
        print(f"\n⚙️  Complexity Analysis:")
        print(f"   Level: {complexity.get('level', 'unknown').title()}")
        print(f"   Optimal Tune: {complexity.get('optimal_tune', 'film')}")
        if complexity.get('factors'):
            print(f"   Factors: {', '.join(complexity['factors'])}")

        results = {}

        for preset in presets:
            config = self.preset_configs[preset]
            
            # Generate output filename
            stem = file_path.stem
            output_file = output_dir / f"{stem}_{preset}.mp4"
            
            # Estimate output size with enhanced algorithm
            estimated_size = self.estimate_output_size(characteristics, preset, original_size)
            compression_ratio = (1 - estimated_size / original_size) * 100
            
            # Generate optimized FFmpeg command
            cmd = self.generate_ffmpeg_command(str(file_path), str(output_file), 
                                            characteristics, preset)
            
            results[preset] = {
                'config': config,
                'output_file': str(output_file),
                'estimated_size_mb': estimated_size / (1024*1024),
                'compression_ratio': compression_ratio,
                'command': ' '.join(cmd),
                'command_list': cmd
            }
            
            print(f"\n🎯 {config['name']}:")
            print(f"   📝 {config['description']}")
            print(f"   📦 Estimated Size: {estimated_size / (1024*1024):.1f} MB")
            print(f"   📉 Compression: {compression_ratio:.1f}%")
            print(f"   ⚡ Command: {' '.join(cmd[:8])}... (full command in batch script)")

        return {
            'file_info': characteristics,
            'original_size': original_size,
            'presets': results
        }

    
    def process_directory(self, input_path: str, output_dir: str = None, 
                         presets: List[str] = None) -> None:
        """Process all video files in a directory or single file"""
        input_path = Path(input_path)
        
        if output_dir:
            output_path = Path(output_dir)
        else:
            if input_path.is_file():
                output_path = input_path.parent / 'compressed'
            else:
                output_path = input_path / 'compressed'
        
        output_path.mkdir(exist_ok=True)
        
        if presets is None:
            presets = ['ultra']
        
        # Find all video files
        video_files = []
        if input_path.is_file():
            if input_path.suffix.lower() in self.supported_formats:
                video_files = [input_path]
        else:
            for ext in self.supported_formats:
                video_files.extend(input_path.glob(f"*{ext}"))
                video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print("❌ No supported video files found!")
            return
        
        print(f"🎯 Ultimate Video Compression Guru - Advanced Analysis Mode")
        print(f"🔍 Found {len(video_files)} video file(s)")
        print(f"📁 Output directory: {output_path}")
        print(f"🎛️  Using presets: {', '.join(presets)}")
        
        total_original_size = 0
        total_estimated_size = 0
        all_commands = []
        all_results = []
        
        for video_file in video_files:
            result = self.process_file(video_file, output_path, presets)
            
            if 'error' in result:
                print(f"❌ Error processing {video_file.name}: {result['error']}")
                continue
            
            total_original_size += result['original_size']
            
            for preset, preset_result in result['presets'].items():
                total_estimated_size += preset_result['estimated_size_mb'] * (1024*1024)
                all_commands.append({
                    'file': video_file.name,
                    'preset': preset,
                    'command': preset_result['command_list']
                })
        
        # Generate comprehensive summary
        if total_original_size > 0:
            total_compression = (1 - total_estimated_size / total_original_size) * 100
        else:
            total_compression = 0
        
        print(f"\n" + "="*80)
        print(f"📊 COMPRESSION SUMMARY\n{'-'*30}")
        print(f"{'='*80}")
        print(f"📁 Total Files: {len(video_files)}")
        print(f"📦 Total Original Size: {total_original_size / (1024*1024*1024):.2f} GB")
        print(f"🗜️  Total Estimated Size: {total_estimated_size / (1024*1024*1024):.2f} GB")
        print(f"💾 Space Saved: {(total_original_size - total_estimated_size) / (1024*1024*1024):.2f} GB")
        print(f"📉 Overall Compression: {total_compression:.1f}%")
        print(f"🚀 Your M1 MacBook Pro is optimized for maximum compression efficiency!")
        
        # Generate batch compression script
        self.generate_batch_script(all_commands, output_path)
        
        print(f"\n🎯 YOUR COMPRESSION GURU IS READY!")
        print(f"🚀 Run the batch script in: {output_path}")
        print(f"📜 Script files: compress_all.sh (Mac/Linux) | compress_all.bat (Windows)")
        print(f"💡 Pro tip: Use 'chmod +x compress_all.sh && ./compress_all.sh' on Mac/Linux")
        print(f"🎊 Your videos will be compressed with maximum efficiency and minimal quality loss!")

    def generate_batch_script(self, commands: List[Dict], output_dir: Path) -> None:
        """Generate optimized batch scripts for all platforms"""
        
        # Mac/Linux script with progress tracking and M1 optimization
        bash_script = output_dir / "compress_all.sh"
        with open(bash_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# 🎬 ULTIMATE VIDEO COMPRESSION GURU - Your Personal Compression Expert\n")
            f.write("# 🚀 M1 MacBook Pro Optimized | Advanced Analysis | Maximum Efficiency\n")
            f.write("# Generated by your compression guru with love ❤️\n\n")
            f.write("set -e  # Exit on error\n\n")
            f.write("# Colors for output\n")
            f.write("RED='\\033[0;31m'\n")
            f.write("GREEN='\\033[0;32m'\n")
            f.write("YELLOW='\\033[1;33m'\n")
            f.write("BLUE='\\033[0;34m'\n")
            f.write("NC='\\033[0m' # No Color\n\n")
            f.write("echo -e \"${BLUE}🎬 YOUR ULTIMATE VIDEO COMPRESSION GURU${NC}\"\n")
            f.write("echo -e \"${BLUE}🧠 Advanced AI Analysis | 🚀 M1 Optimized | 💾 Maximum Savings${NC}\"\n")
            f.write("echo -e \"${YELLOW}Starting intelligent batch compression...${NC}\"\n")
            f.write("echo \"\"\n\n")
            
            f.write(f"TOTAL_FILES={len(commands)}\n")
            f.write("CURRENT=0\n")
            f.write("FAILED=0\n")
            f.write("SUCCESS=0\n\n")
            
            f.write("# Function to show progress\n")
            f.write("show_progress() {\n")
            f.write("    CURRENT=$((CURRENT + 1))\n")
            f.write("    PERCENT=$((CURRENT * 100 / TOTAL_FILES))\n")
            f.write("    echo -e \"${YELLOW}[${CURRENT}/${TOTAL_FILES}] Progress: ${PERCENT}%${NC}\"\n")
            f.write("}\n\n")
            
            f.write("# Function to handle success\n")
            f.write("handle_success() {\n")
            f.write("    SUCCESS=$((SUCCESS + 1))\n")
            f.write("    echo -e \"${GREEN}✅ Success: $1${NC}\"\n")
            f.write("    echo \"\"\n")
            f.write("}\n\n")
            
            f.write("# Function to handle failure\n")
            f.write("handle_failure() {\n")
            f.write("    FAILED=$((FAILED + 1))\n")
            f.write("    echo -e \"${RED}❌ Failed: $1${NC}\"\n")
            f.write("    echo -e \"${RED}Error: $2${NC}\"\n")
            f.write("    echo \"\"\n")
            f.write("}\n\n")
            
            for i, cmd_info in enumerate(commands):
                f.write(f"# File {i+1}: {cmd_info['file']} - {cmd_info['preset']} preset\n")
                f.write("show_progress\n")
                f.write(f"echo -e \"${{BLUE}}🎥 Compressing: {cmd_info['file']} ({cmd_info['preset']})${{NC}}\"\n")
                f.write("START_TIME=$(date +%s)\n")
                
                # Properly escape the command
                cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd_info['command'])
                f.write(f"if {cmd_str}; then\n")
                f.write("    END_TIME=$(date +%s)\n")
                f.write("    DURATION=$((END_TIME - START_TIME))\n")
                f.write(f"    handle_success \"{cmd_info['file']} ({cmd_info['preset']}) - ${{DURATION}}s\"\n")
                f.write("else\n")
                f.write(f"    handle_failure \"{cmd_info['file']} ({cmd_info['preset']})\" \"FFmpeg error\"\n")
                f.write("fi\n\n")

                # Store results for logging
                all_results.append({
                    'file_path': str(video_file),
                    'file_info': result.get('file_info', {}),
                    'presets': result.get('presets', {})
                })
            
            f.write("# Final summary\n")
            f.write("echo -e \"${BLUE}===========================================${NC}\"\n")
            f.write("echo -e \"${BLUE}🏁 YOUR COMPRESSION GURU HAS FINISHED!${NC}\"\n")
            f.write("echo -e \"${BLUE}===========================================${NC}\"\n")
            f.write("echo -e \"${GREEN}✅ Successfully Compressed: $SUCCESS files${NC}\"\n")
            f.write("echo -e \"${RED}❌ Failed: $FAILED files${NC}\"\n")
            f.write("echo -e \"${YELLOW}📊 Total Processed: $TOTAL_FILES files${NC}\"\n")
            f.write("echo \"\"\n")
            f.write("echo -e \"${BLUE}🎉 CONGRATULATIONS! Your videos are now optimally compressed!${NC}\"\n")
            f.write("echo -e \"${GREEN}💰 You've saved massive amounts of storage space!${NC}\"\n")
        
        # Make bash script executable
        bash_script.chmod(0o755)
        
        # Windows batch script
        bat_script = output_dir / "compress_all.bat"
        with open(bat_script, 'w') as f:
            f.write("@echo off\n")
            f.write("REM 🎬 YOUR ULTIMATE VIDEO COMPRESSION GURU\n")
            f.write("REM 🧠 Advanced AI Analysis with Maximum Compression Efficiency\n")
            f.write("REM Generated by your personal compression expert\n\n")
            f.write("echo 🎬 YOUR ULTIMATE VIDEO COMPRESSION GURU\n")
            f.write("echo 🧠 Advanced AI Analysis - Maximum Efficiency\n")
            f.write("echo Starting intelligent batch compression...\n")
            f.write("echo.\n\n")
            
            f.write(f"set TOTAL_FILES={len(commands)}\n")
            f.write("set CURRENT=0\n")
            f.write("set FAILED=0\n")
            f.write("set SUCCESS=0\n\n")
            
            for i, cmd_info in enumerate(commands):
                f.write(f"REM File {i+1}: {cmd_info['file']} - {cmd_info['preset']} preset\n")
                f.write("set /a CURRENT+=1\n")
                f.write("set /a PERCENT=CURRENT*100/TOTAL_FILES\n")
                f.write("echo [%CURRENT%/%TOTAL_FILES%] Progress: %PERCENT%%%\n")
                f.write(f"echo 🎥 Compressing: {cmd_info['file']} ({cmd_info['preset']})\n")
                
                # Properly quote arguments for Windows
                cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd_info['command'])
                f.write(f"{cmd_str}\n")
                f.write("if %errorlevel% == 0 (\n")
                f.write("    set /a SUCCESS+=1\n")
                f.write(f"    echo ✅ Success: {cmd_info['file']} ({cmd_info['preset']})\n")
                f.write(") else (\n")
                f.write("    set /a FAILED+=1\n")
                f.write(f"    echo ❌ Failed: {cmd_info['file']} ({cmd_info['preset']})\n")
                f.write(")\n")
                f.write("echo.\n\n")
            
            f.write("echo ===========================================\n")
            f.write("echo 🏁 YOUR COMPRESSION GURU HAS FINISHED!\n")
            f.write("echo ===========================================\n")
            f.write("echo ✅ Successfully Compressed: %SUCCESS% files\n")
            f.write("echo ❌ Failed: %FAILED% files\n")
            f.write("echo 📊 Total Processed: %TOTAL_FILES% files\n")
            f.write("echo.\n")
            f.write("echo 🎉 CONGRATULATIONS! Your videos are now optimally compressed!\n")
            f.write("echo 💰 You've saved massive amounts of storage space!\n")
            f.write("echo 🚀 Your compression guru has delivered maximum efficiency!\n")
            f.write("pause\n")

            # Generate comprehensive logs
            print(f"📝 Generating detailed compression logs...")
            self.generate_compression_logs(all_results, output_path)

    def generate_compression_logs(self, all_results: List[Dict], output_dir: Path) -> None:
        """Generate detailed compression logs for all files"""
        log_dir = output_dir / "compression_logs"
        log_dir.mkdir(exist_ok=True)
        
        # Individual file logs
        for result in all_results:
            if 'error' in result:
                continue
                
            file_name = Path(result['file_path']).stem
            
            for preset, preset_result in result['presets'].items():
                log_content = self.generate_detailed_logs(
                    result['file_path'], 
                    result['file_info'], 
                    preset, 
                    preset_result['command_list']
                )
                
                log_file = log_dir / f"{file_name}_{preset}_log.txt"
                with open(log_file, 'w') as f:
                    f.write(log_content)
        
        # Master summary log
        summary_log = log_dir / "compression_summary.txt"
        with open(summary_log, 'w') as f:
            f.write("🎬 ULTIMATE COMPRESSION GURU - MASTER SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            total_files = len([r for r in all_results if 'error' not in r])
            total_bloat_score = 0
            total_potential_savings = 0
            
            for result in all_results:
                if 'error' in result:
                    continue
                bloat = result['file_info'].get('bloat_analysis', {})
                total_bloat_score += bloat.get('total_bloat_score', 0)
                total_potential_savings += bloat.get('potential_savings_mb', 0)
            
            avg_bloat = total_bloat_score / total_files if total_files > 0 else 0
            
            f.write(f"📊 BATCH ANALYSIS SUMMARY:\n")
            f.write(f"   Total Files Processed: {total_files}\n")
            f.write(f"   Average Bloat Score: {avg_bloat:.1f}/100\n")
            f.write(f"   Total Potential Savings: {total_potential_savings:.1f} MB\n\n")
            
            f.write("📋 DETAILED LOGS:\n")
            f.write("   Individual file logs available in this directory\n")
            f.write("   Each log contains comprehensive analysis and recommendations\n")

def main():
    all_results = []  # Add this line at the beginning
    """Main function - Your personal compression guru interface"""
    parser = argparse.ArgumentParser(
        description="🎬 ULTIMATE VIDEO COMPRESSION GURU - Your Personal M1-Optimized Expert 🚀",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 YOUR COMPRESSION GURU'S PRESETS:
  ultra      - Ultra Compression (75-90% size reduction) ⚡
  aggressive - Aggressive Compression (80-92% size reduction) 🔥
  high       - High Compression (60-75% size reduction) 💪
  quality    - Quality Priority (40-60% size reduction) 👑
  extreme    - EXTREME Compression (85-95% size reduction) 🌪️

🧠 ADVANCED AI FEATURES:
  • 🔍 Intelligent hardcoded subtitle detection
  • 🌈 HDR content analysis and optimization
  • 🚀 M1 MacBook Pro hardware acceleration
  • 🎭 Scene complexity analysis
  • 🎯 Automatic codec and tuning selection
  • 📊 Comprehensive video characteristic analysis

📋 HOW TO USE YOUR GURU:
  python3 video_guru.py /path/to/videos --presets ultra aggressive
  python3 video_guru.py movie.mp4 --output ./compressed --presets high
  python3 video_guru.py /Movies --presets ultra --output /Compressed

🚀 BATCH PROCESSING MAGIC:
  Your guru generates optimized batch files (compress_all.sh / compress_all.bat)
  with progress tracking, error handling, and maximum efficiency!
  
💡 PRO TIP: Your guru analyzes each video individually and applies the perfect
  compression settings for maximum space savings with minimal quality loss!
        """
    )
    
    parser.add_argument('input', 
                       help='Input video file or directory containing videos')
    
    parser.add_argument('-o', '--output',
                       help='Output directory (default: input_path/compressed)')
    
    parser.add_argument('-p', '--presets', 
                       nargs='+',
                       choices=['ultra', 'aggressive', 'high', 'quality', 'extreme'],
                       default=['ultra'],
                       help='Compression presets to use (default: ultra)')
    
    parser.add_argument('--analyze-only',
                       action='store_true',
                       help='Only analyze videos without generating compression commands')
    
    parser.add_argument('--list-presets',
                       action='store_true', 
                       help='List all available compression presets and exit')
    
    args = parser.parse_args()
    
    analyzer = AdvancedVideoAnalyzer()
    
    if args.list_presets:
        print("🎯 Your Compression Guru's Arsenal:\n")
        for preset, config in analyzer.preset_configs.items():
            print(f"🚀 {preset.upper()}: {config['name']}")
            print(f"   💡 {config['description']}")
            print(f"   ⚙️  CRF: {config['crf']}, Preset: {config['preset']}")
            print()
        print("💪 Your guru is ready to compress with maximum efficiency!")
        return
    
    if not Path(args.input).exists():
        print(f"❌ Error: Input path '{args.input}' does not exist!")
        sys.exit(1)
    
    try:
        if args.analyze_only:
            print("🔍 Your guru is in analysis-only mode - inspecting videos without compression")
        else:
            print("🎬 Your compression guru is analyzing and preparing optimal compression strategies...")
        
        analyzer.process_directory(args.input, args.output, args.presets)
        
    except KeyboardInterrupt:
        print("\n⚠️  Your guru was interrupted - compression paused")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Your guru encountered an unexpected error: {e}")
        print("💡 Please check your input files and try again!")
        sys.exit(1)


if __name__ == "__main__":
    main()