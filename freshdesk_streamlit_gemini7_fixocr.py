import streamlit as st
import requests
import json
import time
import os
import pickle
import pandas as pd
import tempfile
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from google import genai
from collections import Counter
import PyPDF2
import docx
import io

# --- Dataclasses and Cached OCR Model (Updated) ---
@dataclass
class FreshdeskConfig:
    domain: str
    api_key: str

@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str

@st.cache_resource
def get_ocr_model(ocr_available):
    """Initialize PaddleOCR with new API"""
    if ocr_available:
        from paddleocr import PaddleOCR
        
        try:
            # Use mobile models for better performance in Streamlit
            ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device='cpu',  # Force CPU usage for stability
                lang='en'
            )
            st.success("✅ OCR model (PP-OCRv5 mobile) loaded successfully!")
            return ocr
        except Exception as e:
            st.warning(f"⚠️ Failed to load mobile models: {e}")
            st.info("🔄 Trying default models...")
            
            # Fallback to default models
            try:
                ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device='cpu',
                    lang='en'
                )
                st.success("✅ OCR model (default) loaded successfully!")
                return ocr
            except Exception as e2:
                st.error(f"❌ OCR model loading failed completely: {e2}")
                return None
    return None

# --- Vector Database and Analyzer Classes ---
class VectorDatabase:
    def __init__(self, dimension: int = 384, faiss_available=False, ocr_available=False):
        if not faiss_available: return
        from sentence_transformers import SentenceTransformer
        import faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.ticket_ids = []
        self.ticket_texts = []
        self.ticket_subjects = []
        self.ticket_attachment_counts = []
        self.ticket_attachment_types = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.ocr_model = get_ocr_model(ocr_available)

    def add_tickets(self, tickets: List[Dict], freshdesk_config: FreshdeskConfig):
        total_attachments, processed_attachments = 0, 0
        all_attachment_logs = []

        st.write(f"Adding {len(tickets)} tickets to vector database...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize log file with timestamp and clear previous content
        log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open("ocr_extracted_text.log", "w", encoding="utf-8") as f:
            f.write(f"=== Attachment Processing Log - {log_timestamp} ===\n\n")

        for i, ticket in enumerate(tickets):
            searchable_text = self._create_searchable_text(ticket)
            attachment_text = ""
            attachment_count = 0
            attachment_types_set = set()

            if ticket.get('conversations'):
                for conv in ticket['conversations']:
                    if conv.get('attachments'):
                        for attachment in conv['attachments']:
                            total_attachments += 1
                            attachment_count += 1
                            current_attachment_type = attachment.get('content_type', 'unknown')
                            attachment_types_set.add(current_attachment_type)
                            
                            # Process all attachments with enhanced logging
                            if 'image' in current_attachment_type.lower():
                                extracted_text, log_message = self._extract_image_text_paddle_new(attachment, freshdesk_config)
                            else:
                                extracted_text = self._extract_text_from_attachment(attachment, freshdesk_config)
                                log_message = f"Log: Non-image attachment processed. Type: {current_attachment_type}"
                                if extracted_text:
                                    log_message += f" - Successfully extracted {len(extracted_text)} characters"
                                else:
                                    log_message += " - No text extracted"
                            
                            # Create comprehensive log entry for ALL attachments
                            log_entry = (
                                f"--- Ticket ID: {ticket['id']}, Attachment: {attachment.get('name', 'Unknown')} ---\n"
                                f"Type: {current_attachment_type}\n"
                                f"{log_message}\n"
                                f"Extracted text preview: {extracted_text[:200]}...\n\n"
                            )
                            all_attachment_logs.append(log_entry)
                            
                            # Write to log immediately to avoid memory issues and ensure persistence
                            with open("ocr_extracted_text.log", "a", encoding="utf-8") as f:
                                f.write(log_entry)
                            
                            if extracted_text:
                                attachment_text += f"\nAttachment ({attachment.get('name', 'Unknown')}): {extracted_text[:1500]}"
                                processed_attachments += 1
            
            full_searchable_text = searchable_text + attachment_text
            embedding = self.encoder.encode([full_searchable_text])
            import faiss
            faiss.normalize_L2(embedding.astype('float32'))
            self.index.add(embedding.astype('float32'))
            self.ticket_ids.append(ticket['id'])
            self.ticket_texts.append(full_searchable_text)
            self.ticket_subjects.append(ticket.get('subject', 'No Subject'))
            self.ticket_attachment_counts.append(attachment_count)
            self.ticket_attachment_types.append(', '.join(sorted(list(attachment_types_set))))
            
            progress = (i + 1) / len(tickets)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i + 1}/{len(tickets)} tickets")

        # Always show log info, even if no OCR specifically
        if all_attachment_logs:
            st.info(f"ℹ️ Detailed attachment processing results have been saved to `ocr_extracted_text.log` ({len(all_attachment_logs)} attachments processed).")
        else:
            st.info("ℹ️ No attachments found in the processed tickets.")
        
        st.success(f"✅ Vector database created with {len(self.ticket_ids)} tickets.")
        st.info(f"📎 Total attachments found: {total_attachments}. Processed: {processed_attachments}.")
        return total_attachments, processed_attachments

    def _resize_image_if_large(self, content: bytes, max_pixels=5_000_000) -> bytes:
        """Resize image if it's too large for OCR processing"""
        try:
            from PIL import Image
            
            # Load image from bytes
            with io.BytesIO(content) as img_buffer:
                img = Image.open(img_buffer)
                width, height = img.size
                total_pixels = width * height
                
                if total_pixels > max_pixels:
                    # Calculate new size
                    scale_factor = (max_pixels / total_pixels) ** 0.5
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Resize image
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Convert back to bytes
                    output_buffer = io.BytesIO()
                    resized_img.save(output_buffer, format='PNG', quality=95)
                    return output_buffer.getvalue()
                
                return content  # Return original if not too large
                
        except Exception as e:
            st.warning(f"Failed to resize image: {e}")
            return content  # Return original on error

    def _extract_image_text_paddle_new(self, attachment: Dict, freshdesk_config: FreshdeskConfig) -> Tuple[str, str]:
        """Extract text using the new working PaddleOCR API"""
        try:
            attachment_url = attachment.get('attachment_url')
            if not attachment_url:
                return "", "Log: Attachment URL was missing."

            auth = (freshdesk_config.api_key, 'X')
            response = requests.get(attachment_url, auth=auth)
            if response.status_code != 200:
                return "", f"Log: Failed to download image. Status code: {response.status_code}"
            
            content = response.content
            
            if not self.ocr_model:
                return "", "Log: OCR model not available"
            
            try:
                # Resize image if too large (prevents hanging)
                processed_content = self._resize_image_if_large(content)
                
                # Create temporary file since new API expects file path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(processed_content)
                    temp_file_path = temp_file.name
                
                try:
                    # Use the working API call from your script
                    result = self.ocr_model.predict(input=temp_file_path)
                    
                    if result:
                        # Extract text using the working method from your script
                        all_text = []
                        for res in result:
                            try:
                                json_data = res.json
                                if 'res' in json_data and 'rec_texts' in json_data['res']:
                                    rec_texts = json_data['res']['rec_texts']
                                    all_text.extend([text.strip() for text in rec_texts if text.strip()])
                            except Exception as e:
                                continue  # Skip this result if parsing fails
                        
                        extracted_text = " ".join(all_text)
                        
                        if not extracted_text:
                            log_message = f"Log: OCR ran successfully but found no readable text. Results: {len(result)} objects processed"
                            return "", log_message
                        else:
                            log_message = f"Log: OCR extracted text from {len(result)} result object(s), {len(extracted_text)} characters total"
                            return extracted_text, log_message
                    else:
                        log_message = f"Log: OCR found no text regions in image. Result was empty."
                        return "", log_message
                        
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass  # Ignore cleanup errors
                        
            except Exception as e:
                log_message = f"Log: OCR processing failed: {type(e).__name__} - {str(e)}"
                return "", log_message
                
        except Exception as e:
            log_message = f"Log: OCR process failed with an exception: {type(e).__name__} - {str(e)}"
            return "", log_message

    def _extract_text_from_attachment(self, attachment: Dict, freshdesk_config: FreshdeskConfig) -> str:
        # This function handles non-image attachments
        try:
            content_type = attachment.get('content_type', '').lower()
            if 'image' in content_type and self.ocr_model:
                text, _ = self._extract_image_text_paddle_new(attachment, freshdesk_config)
                return text

            # Standard processing for other file types
            attachment_url = attachment.get('attachment_url')
            if not attachment_url: return ""
            auth = (freshdesk_config.api_key, 'X')
            response = requests.get(attachment_url, auth=auth)
            if response.status_code != 200: return ""
            content = response.content

            if 'pdf' in content_type:
                return self._extract_pdf_text(content)
            elif 'word' in content_type or 'docx' in content_type:
                return self._extract_docx_text(content)
            elif 'text' in content_type:
                return content.decode('utf-8', errors='ignore')
            else:
                return f"File: {attachment.get('name', 'Unknown')} ({content_type})"
        except Exception as e:
            st.warning(f"Could not process attachment {attachment.get('name', 'Unknown')}: {e}")
            return ""

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            with io.BytesIO(content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file, strict=False)
                return "".join(page.extract_text() + "\n" for page in reader.pages)[:2000]
        except Exception: return ""

    def _extract_docx_text(self, content: bytes) -> str:
        try:
            with io.BytesIO(content) as docx_file:
                doc = docx.Document(docx_file)
                return "".join(p.text + "\n" for p in doc.paragraphs)[:2000]
        except Exception: return ""

    def _create_searchable_text(self, ticket: Dict) -> str:
        text_parts = [
            f"Subject: {ticket.get('subject', '')}",
            f"Description: {ticket.get('description_text', ticket.get('description', ''))}",
            f"Type: {ticket.get('type', '')}",
            f"Tags: {', '.join(ticket.get('tags', []))}"
        ]
        for conv in ticket.get('conversations', [])[:3]:
            body = conv.get('body_text', conv.get('body', ''))
            if body: text_parts.append(f"Conversation: {body[:200]}")
        return " ".join(text_parts)

    def find_similar_tickets(self, query_ticket_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        try:
            query_idx = self.ticket_ids.index(query_ticket_id)
            query_vector = self.index.reconstruct(query_idx).reshape(1, -1)
            similarities, indices = self.index.search(query_vector, top_k + 1)
            return [(self.ticket_ids[idx], float(sim)) for sim, idx in zip(similarities[0], indices[0]) if idx != query_idx][:top_k]
        except ValueError:
            st.error(f"Ticket {query_ticket_id} not found in vector database.")
            return []

    def save_database(self, filepath: str = "ticket_vector_db"):
        import faiss
        faiss.write_index(self.index, f"{filepath}.index")
        metadata = {
            'ticket_ids': self.ticket_ids,
            'ticket_texts': self.ticket_texts,
            'ticket_subjects': self.ticket_subjects,
            'dimension': self.dimension,
            'ticket_attachment_counts': self.ticket_attachment_counts,
            'ticket_attachment_types': self.ticket_attachment_types,
        }
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        st.success(f"✅ Vector database saved to {filepath}")

    def load_database(self, filepath: str = "ticket_vector_db"):
        if not os.path.exists(f"{filepath}.index"): return False
        try:
            import faiss
            self.index = faiss.read_index(f"{filepath}.index")
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            self.ticket_ids = metadata['ticket_ids']
            self.ticket_texts = metadata['ticket_texts']
            self.ticket_subjects = metadata.get('ticket_subjects', ['Unknown'] * len(self.ticket_ids))
            self.dimension = metadata['dimension']
            num_tickets = len(self.ticket_ids)
            self.ticket_attachment_counts = metadata.get('ticket_attachment_counts', [0] * num_tickets)
            self.ticket_attachment_types = metadata.get('ticket_attachment_types', ['N/A'] * num_tickets)
            mod_time = datetime.fromtimestamp(os.path.getmtime(f"{filepath}.index")).strftime('%Y-%m-%d %H:%M:%S')
            st.success(f"✅ Vector DB loaded with {len(self.ticket_ids)} tickets. Last updated: {mod_time}")
            return True
        except FileNotFoundError:
            return False

class ImprovedAggregateAnalysis:
    """Better approaches for aggregate analysis using vector database"""
    
    def __init__(self, vector_db, analyzer):
        self.vector_db = vector_db
        self.analyzer = analyzer
    
    def approach_1_semantic_search_analysis(self, query: str, max_tickets: int = 15) -> str:
        """
        Approach 1: Use semantic search to find relevant tickets, then analyze
        More efficient than map-reduce - leverages vector similarities
        """
        try:
            # Create query embedding
            query_embedding = self.vector_db.encoder.encode([query])
            import faiss
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search for most relevant tickets
            similarities, indices = self.vector_db.index.search(
                query_embedding.astype('float32'), 
                min(max_tickets, len(self.vector_db.ticket_ids))
            )
            
            # Get relevant tickets data - use lower threshold for more inclusive results
            relevant_tickets = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if similarity > 0.1:  # Lower threshold to include more tickets
                    ticket_id = self.vector_db.ticket_ids[idx]
                    ticket_text = self.vector_db.ticket_texts[idx]
                    ticket_subject = self.vector_db.ticket_subjects[idx]
                    relevant_tickets.append({
                        'id': ticket_id,
                        'text': ticket_text,
                        'subject': ticket_subject,
                        'similarity': float(similarity)
                    })
            
            if not relevant_tickets:
                # If no tickets meet threshold, take top 5 regardless
                relevant_tickets = []
                for idx, similarity in zip(indices[0][:5], similarities[0][:5]):
                    ticket_id = self.vector_db.ticket_ids[idx]
                    ticket_text = self.vector_db.ticket_texts[idx]
                    ticket_subject = self.vector_db.ticket_subjects[idx]
                    relevant_tickets.append({
                        'id': ticket_id,
                        'text': ticket_text,
                        'subject': ticket_subject,
                        'similarity': float(similarity)
                    })
            
            if not relevant_tickets:
                return "No tickets found for analysis."
            
            # Create combined context from relevant tickets
            context_parts = []
            for ticket in relevant_tickets:
                context_parts.append(
                    f"Ticket #{ticket['id']} (Similarity: {ticket['similarity']:.3f}):\n"
                    f"Subject: {ticket['subject']}\n"
                    f"Content: {ticket['text'][:800]}...\n"
                )
            
            combined_context = "\n---\n".join(context_parts)
            
            # Single LLM call for aggregate analysis
            analysis_prompt = f"""
            You are analyzing {len(relevant_tickets)} tickets that are semantically related to the query: "{query}"
            
            Perform an AGGREGATE ANALYSIS across these tickets to identify:
            
            1. **Common Patterns & Themes**: What similar issues, topics, or problems appear across multiple tickets?
            2. **Frequency Analysis**: Which types of issues are most common? How often do certain problems occur?
            3. **Resolution Patterns**: What are the typical ways these issues get resolved? Are there standard solutions?
            4. **Trends & Insights**: What trends do you see across these tickets? Any recurring problems or successful strategies?
            5. **Recommendations**: Based on the patterns, what recommendations would you make?
            
            Focus on CROSS-TICKET ANALYSIS rather than individual ticket details.
            
            Ticket Data:
            {combined_context}
            """
            
            result = self.analyzer.analyze_with_llm(combined_context, analysis_prompt)
            
            metadata = f"\n\n📊 **Analysis Metadata:**\n"
            metadata += f"- Analyzed {len(relevant_tickets)} tickets (from {len(self.vector_db.ticket_ids)} total)\n"
            metadata += f"- Similarity range: {min(t['similarity'] for t in relevant_tickets):.3f} - {max(t['similarity'] for t in relevant_tickets):.3f}\n"
            metadata += f"- Ticket IDs: {', '.join(str(t['id']) for t in relevant_tickets)}"
            
            return result + metadata
            
        except Exception as e:
            return f"Error in semantic search analysis: {e}"
    
    def approach_3_pattern_mining(self, selected_ticket_ids: list, query: str) -> str:
        """
        Approach 3: Direct pattern analysis using text mining
        Fast, no LLM calls for initial processing
        """
        try:
            # Collect all text data
            all_texts = []
            ticket_metadata = []
            
            for ticket_id in selected_ticket_ids:
                try:
                    idx = self.vector_db.ticket_ids.index(ticket_id)
                    text = self.vector_db.ticket_texts[idx]
                    subject = self.vector_db.ticket_subjects[idx]
                    all_texts.append(text.lower())
                    ticket_metadata.append({
                        'id': ticket_id,
                        'subject': subject,
                        'text': text
                    })
                except ValueError:
                    continue
            
            if not all_texts:
                return "No valid tickets found for analysis."
            
            # Extract patterns
            patterns = {
                'common_keywords': self._extract_common_keywords(all_texts),
                'error_patterns': self._extract_error_patterns(all_texts),
                'solution_keywords': self._extract_solution_keywords(all_texts),
                'urgency_indicators': self._extract_urgency_indicators(all_texts)
            }
            
            # Create summary of patterns
            pattern_summary = self._create_pattern_summary(patterns, len(all_texts))
            
            # Use LLM for final interpretation
            context = f"""
            Pattern analysis of {len(all_texts)} tickets:
            
            {pattern_summary}
            
            Sample tickets:
            {self._create_sample_context(ticket_metadata[:5])}
            """
            
            interpretation_query = f"""
            Based on the pattern analysis and query "{query}", provide insights about:
            1. Main themes and trends
            2. Common issues and their frequency
            3. Resolution patterns
            4. Recommendations
            """
            
            interpretation = self.analyzer.analyze_with_llm(context, interpretation_query)
            
            return f"{pattern_summary}\n\n**AI Interpretation:**\n{interpretation}"
            
        except Exception as e:
            return f"Error in pattern mining: {e}"
    
    def _extract_common_keywords(self, texts: list) -> list:
        """Extract most common keywords from texts"""
        # Simple keyword extraction
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            all_words.extend([w for w in words if w not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']])
        
        word_counts = Counter(all_words)
        return word_counts.most_common(15)
    
    def _extract_error_patterns(self, texts: list) -> list:
        """Extract common error patterns"""
        error_patterns = []
        error_keywords = ['error', 'failed', 'exception', 'timeout', 'crash', 'bug', 'issue', 'problem']
        
        for text in texts:
            for keyword in error_keywords:
                if keyword in text:
                    # Extract context around error keyword
                    matches = re.finditer(r'\b' + keyword + r'\b', text, re.IGNORECASE)
                    for match in matches:
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        error_patterns.append(context)
        
        return list(set(error_patterns))[:10]  # Unique patterns, limit to 10
    
    def _extract_solution_keywords(self, texts: list) -> list:
        """Extract solution-related keywords"""
        solution_words = []
        solution_keywords = ['resolved', 'fixed', 'solution', 'workaround', 'updated', 'restart', 'reinstall', 'configure']
        
        for text in texts:
            for keyword in solution_keywords:
                if keyword in text:
                    solution_words.append(keyword)
        
        return Counter(solution_words).most_common(10)
    
    def _extract_urgency_indicators(self, texts: list) -> dict:
        """Extract urgency-related patterns"""
        urgency_high = ['urgent', 'critical', 'emergency', 'asap', 'immediately', 'down', 'outage']
        urgency_low = ['minor', 'low priority', 'enhancement', 'feature request']
        
        high_count = sum(1 for text in texts for keyword in urgency_high if keyword in text)
        low_count = sum(1 for text in texts for keyword in urgency_low if keyword in text)
        
        return {
            'high_urgency_indicators': high_count,
            'low_urgency_indicators': low_count,
            'total_tickets': len(texts)
        }
    
    def _create_pattern_summary(self, patterns: dict, total_tickets: int) -> str:
        """Create a readable summary of patterns"""
        summary = f"**Pattern Analysis Summary ({total_tickets} tickets):**\n\n"
        
        summary += "**🔑 Most Common Keywords:**\n"
        for word, count in patterns['common_keywords']:
            summary += f"- {word}: {count} occurrences\n"
        
        summary += f"\n**⚠️ Error Patterns Found:** {len(patterns['error_patterns'])}\n"
        for pattern in patterns['error_patterns'][:5]:  # Show top 5
            summary += f"- ...{pattern}...\n"
        
        summary += "\n**✅ Solution Keywords:**\n"
        for word, count in patterns['solution_keywords']:
            summary += f"- {word}: {count} occurrences\n"
        
        urgency = patterns['urgency_indicators']
        summary += f"\n**🚨 Urgency Analysis:**\n"
        summary += f"- High urgency indicators: {urgency['high_urgency_indicators']}\n"
        summary += f"- Low urgency indicators: {urgency['low_urgency_indicators']}\n"
        summary += f"- Urgency ratio: {urgency['high_urgency_indicators']/total_tickets:.2%}\n"
        
        return summary
    
    def _create_sample_context(self, ticket_metadata: list) -> str:
        """Create sample context for LLM"""
        sample_context = ""
        for ticket in ticket_metadata:
            sample_context += f"Ticket #{ticket['id']}: {ticket['subject']}\n"
            sample_context += f"{ticket['text'][:300]}...\n\n"
        return sample_context

class FreshdeskTicketAnalyzer:
    def __init__(self, freshdesk_config: FreshdeskConfig, llm_config: LLMConfig, faiss_available=False, ocr_available=False):
        self.freshdesk_config = freshdesk_config
        self.llm_config = llm_config
        self.base_url = f"https://{freshdesk_config.domain}/api/v2"
        self.gemini_client = genai.Client(api_key=llm_config.api_key)
        self.vector_db = VectorDatabase(faiss_available=faiss_available, ocr_available=ocr_available) if faiss_available else None

    @st.cache_data(ttl=3600)
    def get_ticket_by_id(_self, ticket_id: int) -> Dict:
        url = f"{_self.base_url}/tickets/{ticket_id}"
        auth = (_self.freshdesk_config.api_key, 'X')
        try:
            response = requests.get(url, auth=auth)
            response.raise_for_status()
            ticket = response.json()
            ticket['conversations'] = _self.get_conversations(ticket_id)
            return ticket
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching ticket {ticket_id}: {e}")
            return {}

    def get_conversations(self, ticket_id):
        url = f"https://{self.freshdesk_config.domain}/api/v2/tickets/{ticket_id}/conversations"
        auth = (self.freshdesk_config.api_key, 'X')
        r = requests.get(url, auth=auth)
        return r.json() if r.status_code == 200 else []

    @st.cache_data(ttl=600)
    def get_all_tickets(_self, max_tickets: int = None, status_filter: str = "all") -> List[Dict]:
        all_tickets, page = [], 1
        while True:
            tickets_page = _self.get_tickets_page(page, status_filter)
            if not tickets_page: break
            all_tickets.extend(tickets_page)
            if max_tickets and len(all_tickets) >= max_tickets: return all_tickets[:max_tickets]
            if len(tickets_page) < 100: break
            page += 1
            time.sleep(1)
        return all_tickets

    def get_tickets_page(self, page: int, status_filter: str) -> List[Dict]:
        url = f"{self.base_url}/tickets"
        params = {'page': page, 'per_page': 100}
        auth = (self.freshdesk_config.api_key, 'X')
        response = requests.get(url, auth=auth, params=params)
        response.raise_for_status()
        tickets = response.json()
        return [t for t in tickets if t.get('status') == 4] if status_filter == "resolved" else tickets

    def get_enriched_tickets(self, tickets: List[Dict]) -> List[Dict]:
        enriched = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, ticket in enumerate(tickets):
            ticket['conversations'] = self.get_conversations(ticket['id'])
            enriched.append(ticket)
            progress = (i + 1) / len(tickets)
            progress_bar.progress(progress)
            status_text.text(f"Enriching ticket {i + 1}/{len(tickets)}: {ticket['id']}")
            time.sleep(0.1)
        return enriched

    def get_status_name(self, status_code: int) -> str:
        status_map = {2: "Open", 3: "Pending", 4: "Resolved", 5: "Closed"}
        return status_map.get(status_code, f"Status {status_code}")

    def format_ticket_for_analysis(self, ticket: Dict) -> str:
        formatted = f"Ticket ID: {ticket.get('id', 'N/A')}\n"
        formatted += f"Subject: {ticket.get('subject', 'N/A')}\n"
        formatted += f"Status: {self.get_status_name(ticket.get('status', 0))}\n"
        formatted += f"Description: {ticket.get('description_text', 'N/A')}\n"
        if conversations := ticket.get('conversations'):
            formatted += "\nCONVERSATION HISTORY:\n"
            for i, conv in enumerate(conversations):
                body = conv.get('body_text', conv.get('body', 'N/A'))
                formatted += f"\n--- Reply #{i+1} ---\n{body[:1000]}\n"
        return formatted.strip()

    def analyze_with_llm(self, context: str, query: str) -> str:
        prompt = f"""
You are a technical support analyst reviewing Freshdesk tickets. 
Analyze the following ticket data and answer the user's query.
User Query: {query}
Ticket Data:
{context}
Please provide a comprehensive analysis addressing the user's query.
"""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.llm_config.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error analyzing with Gemini: {e}"

    def analyze_multiple_tickets(self, ticket_ids: List[int], query: str) -> str:
        summaries = []
        map_query = f"Based on the high-level question '{query}', extract only the most relevant information from the ticket below. Be very concise (2-3 sentences)."
        status_bar = st.progress(0, text="Starting Map phase...")

        for i, ticket_id in enumerate(ticket_ids):
            try:
                idx = self.vector_db.ticket_ids.index(ticket_id)
                ticket_text = self.vector_db.ticket_texts[idx]
                summary = self.analyze_with_llm(context=ticket_text, query=map_query)
                summaries.append(f"Summary for Ticket #{ticket_id}:\n{summary}\n")
                time.sleep(6)
                progress = (i + 1) / len(ticket_ids)
                status_bar.progress(progress, text=f"Summarizing ticket {ticket_id} ({i+1}/{len(ticket_ids)}). Pausing for rate limit...")
            except Exception as e:
                summaries.append(f"Error processing ticket #{ticket_id}: {e}\n")

        status_bar.progress(1.0, text="Map phase complete. Starting Reduce phase...")
        combined_summaries = "\n---\n".join(summaries)
        reduce_query = f"You are a senior support analyst. Synthesize the following ticket summaries to answer the user's high-level question: '{query}'. Look for patterns, trends, and common themes."
        final_analysis = self.analyze_with_llm(context=combined_summaries, query=reduce_query)
        status_bar.empty()
        return final_analysis


def main():
    st.set_page_config(page_title="Freshdesk Ticket Analyzer", page_icon="🎯", layout="wide")

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        FAISS_AVAILABLE = True
    except ImportError as e:
        st.error(f"FAISS/SentenceTransformers not available. Please run 'pip install faiss-cpu sentence-transformers' and 'pip install --upgrade transformers'. Error: {e}")
        FAISS_AVAILABLE = False

    try:
        from paddleocr import PaddleOCR
        from PIL import Image
        OCR_AVAILABLE = True
        st.success("✅ OCR libraries (PaddleOCR, PIL) are available")
    except ImportError as e:
        st.warning(f"⚠️ OCR libraries not available. Text cannot be extracted from images. Run 'pip install paddlepaddle paddleocr pillow'. Error: {e}")
        OCR_AVAILABLE = False
    
    st.title("🎯 Freshdesk Ticket Analyzer with Gemini")

    try:
        GEMINI_KEY = st.secrets["GEMINI_KEY"]
        FRESHDESK_API_KEY_DEFAULT = st.secrets["FRESHDESK_API_KEY"]
        FRESHDESK_DOMAIN_DEFAULT = st.secrets["FRESHDESK_DOMAIN"]
    except (FileNotFoundError, KeyError):
        st.error("🚨 `.streamlit/secrets.toml` not found or misconfigured.")
        return

    st.sidebar.header("Configuration")
    freshdesk_domain = st.sidebar.text_input("Freshdesk Domain", value=FRESHDESK_DOMAIN_DEFAULT)
    freshdesk_api_key = st.sidebar.text_input("Freshdesk API Key", value=FRESHDESK_API_KEY_DEFAULT, type="password")

    if not all([freshdesk_domain, freshdesk_api_key, GEMINI_KEY]):
        st.warning("Please ensure all configurations in the sidebar are set.")
        return

    if 'analyzer' not in st.session_state:
        freshdesk_config = FreshdeskConfig(domain=freshdesk_domain, api_key=freshdesk_api_key)
        llm_config = LLMConfig(provider="gemini", api_key=GEMINI_KEY, model="gemini-2.5-flash-preview-05-20")
        st.session_state.analyzer = FreshdeskTicketAnalyzer(freshdesk_config, llm_config, faiss_available=FAISS_AVAILABLE, ocr_available=OCR_AVAILABLE)
        if FAISS_AVAILABLE:
            st.session_state.analyzer.vector_db.load_database()

    tabs = ["📊 Vector Database", "🔍 Ticket Analysis", "🔗 Similar Tickets", "📈 Aggregate Analysis"]
    tab1, tab2, tab3, tab4 = st.tabs(tabs)

    with tab1:
        st.header("Vector Database Management")
        if not FAISS_AVAILABLE: st.error("This feature is disabled. Please check the error message at the top of the page."); return
        if OCR_AVAILABLE:
            st.info("ℹ️ OCR (PaddleOCR v5 mobile) is enabled. Text will be extracted from image attachments with automatic resizing for large images.")
        else:
            st.warning("⚠️ OCR is disabled. Text cannot be extracted from images.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Create / Update Database")
            max_tickets = st.number_input("Number of tickets to process", 1, 1000, 50)
            status_filter = st.selectbox("Ticket Status Filter", ["all", "resolved"])
            if st.button("Create Vector Database", type="primary"):
                st.session_state.analyzer.vector_db = VectorDatabase(
                    faiss_available=FAISS_AVAILABLE, 
                    ocr_available=OCR_AVAILABLE
                )
                with st.spinner("Fetching and processing tickets..."):
                    tickets = st.session_state.analyzer.get_all_tickets(max_tickets, status_filter)
                    if tickets:
                        enriched_tickets = st.session_state.analyzer.get_enriched_tickets(tickets)
                        st.session_state.analyzer.vector_db.add_tickets(enriched_tickets, st.session_state.analyzer.freshdesk_config)
                        st.session_state.analyzer.vector_db.save_database()
                    else: st.error("No tickets found!")
        
        with col2:
            st.subheader("Database Status")
            if st.session_state.analyzer.vector_db and st.session_state.analyzer.vector_db.ticket_ids:
                st.metric("Tickets in DB", len(st.session_state.analyzer.vector_db.ticket_ids))
                if st.button("Reload from disk"):
                    st.session_state.analyzer.vector_db.load_database()
            else:
                st.info("No vector database loaded. Create one to enable similarity search.")

        st.markdown("---")
        st.subheader("📋 Tickets in Vector Database")
        if FAISS_AVAILABLE and st.session_state.analyzer.vector_db and st.session_state.analyzer.vector_db.ticket_ids:
            df_data = []
            for ticket_id, subject, count, types in zip(
                st.session_state.analyzer.vector_db.ticket_ids,
                st.session_state.analyzer.vector_db.ticket_subjects,
                st.session_state.analyzer.vector_db.ticket_attachment_counts,
                st.session_state.analyzer.vector_db.ticket_attachment_types
            ):
                df_data.append({
                    'Ticket ID': ticket_id,
                    'Subject': subject[:60] + '...' if len(subject) > 60 else subject,
                    'Num Attachments': count,
                    'Attachment Types': types,
                    'URL': f"https://{freshdesk_domain}/a/tickets/{ticket_id}"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No tickets in the database to display.")


    with tab2:
        st.header("Single Ticket Analysis")
        ticket_id = st.number_input("Ticket ID", min_value=1, key="analysis_ticket_id")
        if st.button("Fetch Ticket", type="primary", disabled=not ticket_id):
            with st.spinner(f"Fetching ticket {ticket_id}..."):
                st.session_state.current_ticket = st.session_state.analyzer.get_ticket_by_id(ticket_id)
        if 'current_ticket' in st.session_state and st.session_state.current_ticket:
            ticket = st.session_state.current_ticket
            st.subheader(f"Analysis for Ticket #{ticket['id']}: {ticket.get('subject', 'N/A')}")
            query = st.text_area("Ask a question about this ticket:", "Summarize the main issue and resolution.")
            if st.button("Analyze with Gemini"):
                with st.spinner("Analyzing..."):
                    context = st.session_state.analyzer.format_ticket_for_analysis(ticket)
                    response = st.session_state.analyzer.analyze_with_llm(context, query)
                    st.markdown(response)
            with st.expander("Full Ticket Details"):
                st.text(st.session_state.analyzer.format_ticket_for_analysis(ticket))

    with tab3:
        st.header("Find Similar Tickets")
        if not FAISS_AVAILABLE or not st.session_state.analyzer.vector_db or not st.session_state.analyzer.vector_db.ticket_ids:
            st.warning("Please create or load a vector database first!"); return
        
        ticket_id_options = st.session_state.analyzer.vector_db.ticket_ids
        if not ticket_id_options:
            st.info("No tickets in the database to select from.")
        else:
            ticket_id = st.selectbox("Find tickets similar to:", options=ticket_id_options, index=None, placeholder="Select a ticket...")
            
            # Show source ticket details
            if ticket_id:
                with st.spinner(f"Loading details for source ticket #{ticket_id}..."):
                    source_ticket_data = st.session_state.analyzer.get_ticket_by_id(ticket_id)
                
                if source_ticket_data:
                    with st.expander(f"Source Ticket Details: #{source_ticket_data.get('id')} - {source_ticket_data.get('subject')}", expanded=True):
                        # Use st.text() instead of st.text_area() to make it copyable
                        formatted_source = st.session_state.analyzer.format_ticket_for_analysis(source_ticket_data)
                        st.text(formatted_source)
                st.markdown("---")

            top_k = st.slider("Number of similar tickets to show", 1, 15, 5)

            if st.button("Find Similar Tickets", type="primary", disabled=not ticket_id):
                st.session_state[f'similar_results_{ticket_id}'] = True
                
            if ticket_id and st.session_state.get(f'similar_results_{ticket_id}'):
                analyzer = st.session_state.analyzer
                
                similar_tickets = analyzer.vector_db.find_similar_tickets(ticket_id, top_k)
                if not similar_tickets:
                    st.warning(f"No similar tickets found for ticket {ticket_id}.")
                else:
                    st.subheader(f"🔍 Top {len(similar_tickets)} Similar Tickets")
                    for i, (similar_id, similarity) in enumerate(similar_tickets, 1):
                        with st.expander(f"#{i}: Ticket #{similar_id} (Similarity: {similarity:.3f})"):
                            ticket_data = analyzer.get_ticket_by_id(similar_id)
                            if ticket_data:
                                ticket_url = f"https://{freshdesk_domain}/a/tickets/{similar_id}"
                                st.write(f"**Subject:** {ticket_data.get('subject', 'N/A')}")
                                st.write(f"**URL:** [{ticket_url}]({ticket_url})")
                                
                                # Use checkbox + st.text() instead of nested text_area to make it copyable
                                if st.checkbox("Show Full Ticket Details", key=f"details_{similar_id}"):
                                    formatted_details = analyzer.format_ticket_for_analysis(ticket_data)
                                    st.text(formatted_details)
                    
                    # Add Analysis of Most Similar Ticket section
                    if similar_tickets:
                        most_similar_id, most_similar_score = similar_tickets[0]
                        most_similar_ticket = analyzer.get_ticket_by_id(most_similar_id)
                        
                        if most_similar_ticket:
                            st.markdown("---")
                            st.markdown("### 🎯 **Analysis of Most Similar Ticket**")
                            st.info(f"Compare your original ticket #{ticket_id} with the most similar ticket #{most_similar_id} (similarity: {most_similar_score:.3f})")
                            
                            analysis_key = f'analysis_{ticket_id}_{most_similar_id}'
                            
                            if st.button("🔍 Analyze & Suggest Solution", key=f"analyze_btn_{ticket_id}", type="secondary"):
                                with st.spinner("Generating solution based on most similar ticket..."):
                                    original_context = analyzer.format_ticket_for_analysis(source_ticket_data)
                                    similar_context = analyzer.format_ticket_for_analysis(most_similar_ticket)
                                    
                                    analysis_query = f"""
                                    Analyze the most similar ticket (#{most_similar_id}) and suggest a resolution for the original ticket (#{ticket_id}). 
                                    
                                    Please provide:
                                    1. Key similarities between the tickets
                                    2. How the similar ticket was resolved
                                    3. Actionable steps to resolve the original ticket
                                    4. Any differences to consider
                                    """
                                    
                                    combined_context = f"ORIGINAL TICKET:\n{original_context}\n\nMOST SIMILAR TICKET:\n{similar_context}"
                                    st.session_state[analysis_key] = analyzer.analyze_with_llm(context=combined_context, query=analysis_query)
                            
                            if analysis_key in st.session_state:
                                st.markdown("#### 📊 **Analysis & Solution Recommendation**")
                                st.markdown(st.session_state[analysis_key])


    with tab4:
        st.header("📈 Intelligent Aggregate Analysis")
        if not FAISS_AVAILABLE or not st.session_state.analyzer.vector_db or not st.session_state.analyzer.vector_db.ticket_ids:
            st.warning("Please create or load a vector database first!")
        else:
            st.info("🚀 **Aggregate Analysis**: These methods analyze patterns across multiple tickets using your vector database!")
            
            # Create improved analyzer instance
            if 'improved_analyzer' not in st.session_state:
                st.session_state.improved_analyzer = ImprovedAggregateAnalysis(
                    st.session_state.analyzer.vector_db, 
                    st.session_state.analyzer
                )
            
            # Analysis method selection
            analysis_method = st.radio(
                "Choose Analysis Method:",
                options=[
                    "🔍 Semantic Search Analysis (Recommended)",
                    "⚡ Pattern Mining + AI",
                    "📊 Classic Map-Reduce (Original)"
                ],
                help="""
                • **Semantic Search**: Finds tickets similar to your query, then analyzes patterns across them (1 LLM call)
                • **Pattern Mining**: Extracts patterns first, then AI interprets them (1 LLM call)
                • **Map-Reduce**: Original method - slower but comprehensive (N+1 LLM calls)
                """
            )
            
            # Query input
            agg_query = st.text_area(
                "What do you want to analyze across your tickets?", 
                "What are the most common bugs and how are they typically resolved?", 
                height=80,
                help="Describe what patterns or trends you want to find across multiple tickets"
            )
            
            if analysis_method == "🔍 Semantic Search Analysis (Recommended)":
                max_tickets = st.slider("Max tickets to analyze", 5, 30, 15)
                st.info("💡 This will find tickets semantically related to your query and analyze patterns across them")
                
                if st.button("🚀 Run Semantic Analysis", type="primary", disabled=not agg_query):
                    with st.spinner("🔍 Finding relevant tickets and analyzing patterns..."):
                        start_time = time.time()
                        result = st.session_state.improved_analyzer.approach_1_semantic_search_analysis(
                            agg_query, max_tickets
                        )
                        elapsed = time.time() - start_time
                        
                        st.success(f"✅ Analysis completed in {elapsed:.1f} seconds!")
                        st.markdown(result)
            
            elif analysis_method == "⚡ Pattern Mining + AI":
                all_ids = st.session_state.analyzer.vector_db.ticket_ids
                selected_ids = st.multiselect(
                    "Select tickets for pattern analysis:", 
                    all_ids, 
                    default=all_ids[:20] if all_ids else [],
                    help="Choose specific tickets to analyze for patterns"
                )
                
                if st.button("⚡ Run Pattern Analysis", type="primary", disabled=not selected_ids):
                    with st.spinner("⚡ Mining patterns and generating AI insights..."):
                        start_time = time.time()
                        result = st.session_state.improved_analyzer.approach_3_pattern_mining(
                            selected_ids, agg_query
                        )
                        elapsed = time.time() - start_time
                        
                        st.success(f"✅ Pattern analysis completed in {elapsed:.1f} seconds!")
                        st.markdown(result)
            
            else:  # Classic Map-Reduce
                all_ids = st.session_state.analyzer.vector_db.ticket_ids
                selected_ids = st.multiselect(
                    "Select tickets for map-reduce analysis:", 
                    all_ids, 
                    default=all_ids[:10] if all_ids else []
                )
                
                st.warning("⚠️ **Slower method**: Makes 1 LLM call per ticket + 1 final call. Use for detailed analysis only.")
                
                if st.button("📊 Run Map-Reduce Analysis", type="primary", disabled=not selected_ids or not agg_query):
                    final_analysis = st.session_state.analyzer.analyze_multiple_tickets(selected_ids, agg_query)
                    st.subheader("📊 Map-Reduce Analysis Result")
                    st.markdown(final_analysis)
            
            # Performance comparison info
            with st.expander("📊 Method Comparison", expanded=False):
                comparison_data = {
                    "Method": [
                        "🔍 Semantic Search", 
                        "⚡ Pattern Mining", 
                        "📊 Map-Reduce"
                    ],
                    "LLM Calls": ["1", "1", "N+1"],
                    "Speed": ["Fast", "Very Fast", "Slow"],
                    "Best For": [
                        "Finding tickets related to specific topics/issues",
                        "Quick statistical insights and keyword analysis", 
                        "Comprehensive deep analysis of selected tickets"
                    ],
                    "Vector DB Usage": ["✅ Full", "✅ Partial", "❌ None"],
                    "Ticket Selection": ["Automatic (by query)", "Manual selection", "Manual selection"]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()