type SignSageLogoProps = {
  className?: string;
};

/** The SignSage brand mark. Single source — navbar, footer, and shell share it. */
export function SignSageLogo({ className = "h-9 w-9" }: SignSageLogoProps) {
  return (
    <img
      src="/logo.png"
      alt="SignSage"
      className={`rounded-lg object-cover ${className}`}
    />
  );
}
