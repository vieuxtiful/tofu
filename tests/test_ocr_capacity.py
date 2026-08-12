import threading
import time


def test_ocr_slot_bounds_concurrent_work(monkeypatch):
    import main

    monkeypatch.setattr(main, "_OCR_SLOTS", threading.BoundedSemaphore(1))
    active = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal active, peak
        with main._ocr_slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    workers = [threading.Thread(target=work) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert peak == 1
