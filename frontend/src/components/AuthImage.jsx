import { useEffect, useState } from "react";
import { fetchImageUrl } from "../api.js";

/** <img> for protected endpoints: fetches with the JWT and shows a blob URL. */
export default function AuthImage({ path, alt = "", className }) {
  const [url, setUrl] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    let obj;
    setUrl(null);
    setFailed(false);
    if (path) {
      fetchImageUrl(path)
        .then((u) => { obj = u; if (live) setUrl(u); else URL.revokeObjectURL(u); })
        .catch(() => live && setFailed(true));
    }
    return () => { live = false; if (obj) URL.revokeObjectURL(obj); };
  }, [path]);

  if (failed) return <div className={`img-placeholder ${className || ""}`}>Unavailable</div>;
  if (!url) return <div className={`img-placeholder shimmer ${className || ""}`} />;
  return <img src={url} alt={alt} className={className} />;
}
