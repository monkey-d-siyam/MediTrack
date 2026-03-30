/**
 * MediTrack — Global JavaScript
 * Handles: toast auto-dismiss, animations, navbar effects
 */

document.addEventListener('DOMContentLoaded', () => {

    // ── Auto-dismiss toast notifications ─────────────────────────────────────
    const toasts = document.querySelectorAll('.toast.show');
    toasts.forEach(toast => {
        const bsToast = new bootstrap.Toast(toast, { delay: 4500 });
        bsToast.show();
    });

    // ── Navbar scroll effect ──────────────────────────────────────────────────
    const nav = document.getElementById('mainNav');
    if (nav) {
        window.addEventListener('scroll', () => {
            nav.style.borderBottomColor = window.scrollY > 20
                ? 'rgba(99,102,241,0.25)'
                : 'rgba(255,255,255,0.08)';
        }, { passive: true });
    }

    // ── Animate progress bars on scroll ──────────────────────────────────────
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = entry.target.dataset.width || entry.target.style.width;
            }
        });
    }, { threshold: 0.3 });

    document.querySelectorAll('.progress-bar').forEach(bar => {
        observer.observe(bar);
    });

    // ── Confirm delete buttons ────────────────────────────────────────────────
    // (handled via inline onsubmit, but could be enhanced here)

});
