"""Test OCR post-processing with ACTUAL production OCR output"""
import sys, os, types, importlib.util
sys.path.insert(0, ".")

import unittest.mock as mock

# Create mock modules
for mod_name in ['app', 'app.rag', 'app.rag.embeddings', 'app.rag.vector_store', 'app.rag.config']:
    sys.modules[mod_name] = types.ModuleType(mod_name)

mock_config = mock.MagicMock()
mock_config.chunk_size = 512
mock_config.chunk_overlap = 50
sys.modules['app.rag.config'].rag_config = mock_config
sys.modules['app.rag.embeddings'].embeddings_service = mock.MagicMock()
sys.modules['app.rag.vector_store'].vector_store = mock.MagicMock()

spec = importlib.util.spec_from_file_location(
    "app.rag.document_processor",
    os.path.join("app", "rag", "document_processor.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DocumentProcessor = module.DocumentProcessor

# ACTUAL OCR output from production (chunk 1 from the API)
ACTUAL_OCR_PAGE = """1. PERSIAPAN KEBUN PERBANYAKAN
11,

1.2.

1.3.

14.

1.5.

1.6.

Definisi
Persiapan kebun perbanyakan adalah kegiatan mempersiapkan blok kebun TM teh
pilihan yang akan dijadikan sebagai kebun perbanyakan

Tujuan
Memperoleh kebun perbanyakan dengan kondisi sebagai berikut:
Tanaman teh yang dipelihara tumbuh sehat dan jagur
Klon tanaman teh yang dipelihara homogen sesuai dengan program penanaman
TTI
Potensi jumlah stekres yang dihasilkan sesuai dengan standar

Sasaran
Mendapatkan kebun perbanyakan dengan luasan, potensi jumlah stekres dan jenis
klon sesuai dengan program penanaman TTI

Prinsip Umum

1) Membuat jadwal kegiatan pekerjaan persiapan kebun perbanyakan sesuai
dengan waktu yang ditentukan.

2) Penentuan lokasi blok kebun perbanyakan dengan mempertimbangkan
topografi, populasi, sumber air dan kemudahan transportasi

3) Mempersiapkan kebun perbanyakan untuk mendapatkan jumlah stekress sesuai
kebutuhan dengan cara pangkasan bersih.

Alat, Bahan dan Sarana

1) Alat pengukur luas areal (GPS)

2) Ajir dan cam label untuk identifikasi klon

3) Alat pangkas selengkapnya ( gaet, gergaji pangkas dan ukuran/meteran )
4) Perlengkapan paket pangkas (cangkul, garpu, gacok, alat gosok lukut dsb.)

Prosedur Kerja

a. Penentuan blok kebun TM Teh yang sesuai dengan kriteria kebun perbanyakan
antara lain:
Sehat dan bebas dari hama penyakit.
Telah mengalami perlakuan pemurnian klon sesuai anjuran.
Dipelihara secara khusus dan telah dipersiapkan setahun sebelum pesemaian
dibuat.
Memiliki luas yang cukup dengan rencana pembuatan pesemaian."""

# Also test chunk 3 pattern
ACTUAL_OCR_PAGE2 = """2.1. Definisi
2.2.

2.3.

2.4,

2.5

2.6.

Pemeliharaan kebun perbanyakan adalah serangkaian kegiatan pemeliharaan
tanaman teh yang dilaksanakan di kebun perbanyakan.

Tujuan
1) Menghasilkan kebun perbanyakan yang sehat, tumbuh dengan mulus dan subur
2) Menghasilkan tunas primer yang sehat dan jagur dengan jumlah yang optimal

Sasaran
Menghasilkan antara 15 - 20 stekres per pohon dengan jumlah stek per stekres
antara 3 - 4 stek

Prinsip Umum

1) Penyiangan dilaksanakan dengan cara kombinasi penyiangan kimia dan manual

2) Pengendalian hama dan penyakit bersifat preventif"""


def main():
    dp = DocumentProcessor()
    
    print("=" * 70)
    print("TEST: OCR Post-Processing on ACTUAL production output")
    print("=" * 70)
    
    print("\n--- BEFORE post-processing (first 30 lines) ---")
    for i, line in enumerate(ACTUAL_OCR_PAGE.split('\n')[:30]):
        print(f"  {i:2d}: {repr(line)}")
    
    fixed = dp._postprocess_ocr_text(ACTUAL_OCR_PAGE)
    
    print("\n--- AFTER post-processing (first 30 lines) ---")
    for i, line in enumerate(fixed.split('\n')[:30]):
        print(f"  {i:2d}: {repr(line)}")
    
    print("\n" + "=" * 70)
    print("TEST: Number fix examples")
    print("=" * 70)
    test_cases = ['11,', '14.', '1.2.', '2.4,', '2.5', '111']
    for tc in test_cases:
        print(f"  {tc!r:10s} -> {dp._fix_ocr_number(tc)!r}")
    
    print("\n" + "=" * 70)
    print("TEST: Section detection on FIXED text")
    print("=" * 70)
    is_structured = dp._is_structured_document(fixed)
    print(f"  Is structured: {is_structured}")
    
    sections = dp._parse_sections(fixed)
    for i, sec in enumerate(sections):
        heading = sec["heading"] or "(preamble)"
        body_preview = sec["body"][:60].replace('\n', ' ') if sec["body"] else "(empty)"
        print(f"  [{i}] L{sec['heading_level']} | {heading}")
        print(f"       Body: {body_preview}")
    
    print("\n" + "=" * 70)
    print("TEST: Page 2 (section 2.x) post-processing")
    print("=" * 70)
    
    print("\n--- BEFORE ---")
    for i, line in enumerate(ACTUAL_OCR_PAGE2.split('\n')[:20]):
        print(f"  {i:2d}: {repr(line)}")
    
    fixed2 = dp._postprocess_ocr_text(ACTUAL_OCR_PAGE2)
    
    print("\n--- AFTER ---")
    for i, line in enumerate(fixed2.split('\n')[:20]):
        print(f"  {i:2d}: {repr(line)}")
    
    print("\n" + "=" * 70)
    print("TEST: Full semantic chunking with post-processed text")
    print("=" * 70)
    pages = [
        {"page_number": 1, "text": fixed},
        {"page_number": 2, "text": fixed2},
    ]
    chunks = dp.chunk_text_semantic_with_pages(pages)
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i} (page {c['page_number']}) ---")
        print(f"  Heading: {c.get('heading')}")
        print(f"  Parent:  {c.get('parent_heading')}")
        preview = c['text'][:200].replace('\n', '\\n')
        print(f"  Text:    {preview}")
        print(f"  Tokens:  {dp.count_tokens(c['text'])}")


if __name__ == "__main__":
    main()
