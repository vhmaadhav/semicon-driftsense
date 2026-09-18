(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const total = slides.length;
  const progress = document.getElementById("progress");
  const hud = document.getElementById("hud-current");
  const hudTotal = document.getElementById("hud-total");
  hudTotal.textContent = total;

  function indexFromHash() {
    const n = parseInt(location.hash.replace("#", ""), 10);
    if (!isNaN(n) && n >= 1 && n <= total) return n - 1;
    return 0;
  }

  let current = indexFromHash();

  function render(prevIndex) {
    slides.forEach((s, i) => {
      s.classList.remove("active", "exit-left");
      if (i === current) {
        s.classList.add("active");
      } else if (prevIndex !== undefined && i === prevIndex && prevIndex < current) {
        s.classList.add("exit-left");
      }
    });
    progress.style.width = ((current + 1) / total) * 100 + "%";
    hud.textContent = current + 1;
    location.hash = String(current + 1);
  }

  function go(delta) {
    const prev = current;
    const next = current + delta;
    if (next < 0 || next >= total) return;
    current = next;
    render(prev);
  }

  window.addEventListener("keydown", (e) => {
    if (["ArrowRight", "PageDown", " "].includes(e.key)) { go(1); e.preventDefault(); }
    else if (["ArrowLeft", "PageUp"].includes(e.key)) { go(-1); e.preventDefault(); }
    else if (e.key === "Home") { current = 0; render(); e.preventDefault(); }
    else if (e.key === "End") { current = total - 1; render(); e.preventDefault(); }
    else if (e.key === "f" || e.key === "F") {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen();
      else document.exitFullscreen();
    }
  });

  window.addEventListener("hashchange", () => {
    const n = indexFromHash();
    if (n !== current) { current = n; render(); }
  });

  let touchStartX = null;
  window.addEventListener("touchstart", (e) => { touchStartX = e.touches[0].clientX; });
  window.addEventListener("touchend", (e) => {
    if (touchStartX === null) return;
    const dx = e.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) > 60) go(dx < 0 ? 1 : -1);
    touchStartX = null;
  });

  document.getElementById("nav-prev").addEventListener("click", () => go(-1));
  document.getElementById("nav-next").addEventListener("click", () => go(1));

  render();
})();
