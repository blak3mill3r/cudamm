#ifndef CUDA_EVENT_HPP
#define CUDA_EVENT_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{
	class Stream;

	/// CUDA events
	/**
		CUDA events are used to keep track, synchronize and time
		asynchronous operations.
	
		Noncopyable.
	*/
	class Event : boost::noncopyable
	{
		public:
			/// Create event
			Event();
			
			/// Destroy event
			~Event();
			
			/// Record event
			/**
				The event will be recorded when all preceding operations in
				the CUDA context have been completed.
			*/
			void record() const;
			
			/// Record event
			/**
				The event will be recorded when all preceding operations in
				the given stream have been completed.
			*/
			void record(const Stream &stream) const;
			
			/// Block until the event has actually been recorded
			/**
				An exception will be thrown if record is not called on
				this event.
			*/
			void synchronize() const;
			
			/// Query if the event has actually been recorded
			/**
				An exception will be thrown if record is not called on
				this event.

				@return true if recorded
			*/
			bool query() const;
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;

			friend float operator-(const Event &end, const Event &start);
	};
	
	float operator-(const Event &end, const Event &start);
}

#endif

